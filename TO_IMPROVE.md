# TO_IMPROVE

Deferred work for generalizing the benchmark to **heterogeneous model output modes**
(binary classifier vs point forecaster vs quantile vs embedding). Background and the
full analysis live in [`crossec_forecast/models/pretrained_research.md`](crossec_forecast/models/pretrained_research.md).

---

## Done in the "generic base" pass

- `BaseClassifierModel` gained `output_kind`, `to_score(raw) -> [B]` (higher = more
  bullish), and `compute_loss(raw, batch) -> scalar` (model owns its loss, sees the whole
  batch). The 3 from-scratch models declare `output_kind = "binary_prob"` and inherit the
  defaults (sigmoid + BCE/Focal on `batch["y"]`).
- `Trainer` is now loss-agnostic: `loss_fn` param / `get_loss_fn` wiring removed;
  `train_epoch` / `validate` call `model.compute_loss`, `validate` / `predict` /
  `run_infer` call `model.to_score` (single forward in `validate` now).
- Loss selection moved out of `TrainConfig` / `TrainSchema` into `model.config`
  (`loss_type` / `focal_gamma` / `focal_alpha`).
- `compute_all_metrics(..., output_kind=)` gates the classification block (AUC / acc /
  P / R / F1) to `output_kind == "binary_prob"`; `benchmark.py` reads those keys with
  `.get(..., nan)`.
- `PanelTimeSeriesDataset`: NaN target **or** NaN forward return now drops the sample in
  non-inference mode (no more `0.0` zero-fill); missing `target_col` / `fwd_ret_col`
  raises instead of silently producing an empty dataset.
- **Invariant:** val mean Rank IC stays the sole early-stop / checkpoint / benchmark-sort
  criterion for every `output_kind`.
- **Per-interpreter backend isolation.** `models/_optional.py` (`require_modules`,
  `ModelDependencyError`, `module_available`); `BaseClassifierModel.REQUIRED_MODULES` +
  `PYTHON_HINT`; registry `is_model_available` / `list_available_models` + `build_model`
  guard; `BenchmarkEngine` scans the roster and skips unavailable entries with a warning;
  `pyproject` extras `[chronos]` / `[moment]`. Wrappers keep heavy imports inside
  `__init__`; models are classified by Python version, not env name. See
  `pretrained_research.md` §2.
- **First two TSFM wrappers, both frozen backbone + linear probe:**
  `chronos_bolt_head_only` (B / point_forecast, `.[chronos]`) and `moment_head_only`
  (A / binary_prob, `.[moment]`). The frozen backbone is held as a **non-submodule**
  attribute (`object.__setattr__` for MOMENT, which is itself an `nn.Module`) so
  `parameters()` / `state_dict()` stay head-only.
- **`PanelTimeSeriesDataset` timestamp-filter fix.** The split filter compared
  `np.datetime64` row values against a set of `pandas.Timestamp` from `TimeSplitter`;
  on some pandas/numpy combos that is always `False` → every sample silently dropped
  (`num_samples=0`). Now canonicalized to int64-ns for datetime columns (int/str tick
  ids unchanged). Surfaced when the `.venv310` interpreter was set up.

---

## C1 — `pred_prob` column name is now a misnomer for non-binary models

`Trainer.predict` / `run_infer` still write a column literally named `pred_prob`; for
`output_kind != "binary_prob"` it holds the raw `to_score` value, not a probability.
`compute_all_metrics` and `SimpleLongShortBacktester` also read `pred_prob` by name.

**Fix:** rename the canonical column to `score`; when `output_kind == "binary_prob"`
also emit `pred_prob` as an alias. Update `metrics.py`, `backtest.py`, `benchmark.py`,
`train_pipeline` / `infer_pipeline` artifact writers, and any downstream analysis +
wandb keys.

## C4 — zero-shot models / `epochs == 0` break the benchmark row

`BenchmarkEngine.run` calls `trainer.fit()` unconditionally. With `epochs == 0` the
`fit` loop never runs, `best_val_rank_ic` stays `-inf`, and the benchmark sort / wandb
summary get `-inf`.

**Fix:** if a model declares itself zero-shot (e.g. `output_kind`-adjacent flag or
`epochs == 0`), skip `fit`, run `validate` once to record `val_mean_rank_ic`, and load
whatever weights the wrapper built in `__init__`.

## C5 — quantile / tuple raw outputs break shape assumptions

`to_score` / `compute_loss` currently assume `raw` is a single tensor. A quantile model
returns `[B, Q]`; a model returning `(point, quantiles)` tuple breaks `loss.backward()`
and `.reshape(-1)`.

**Fix:** define a small raw-output convention (single tensor, or a typed container) that
`to_score` and `compute_loss` both understand; document it in `BaseClassifierModel`.

## C6 — Pattern-B raw-series channel (separate from the `fwd_logret` scalar)

**Partially done**, differently from the original plan below: `data.extra_input_cols` /
`extra_input_pattern` now give `PanelTimeSeriesDataset` a third, opt-in column group,
packed into the *same* `x` tensor right after `cov_cols` (`x[..., feature_dim+cov_dim :
feature_dim+cov_dim+extra_input_dim]`), auto-injected as `self.extra_input_dim` on
`BaseClassifierModel`. `chronos_bolt_zeroshot` is the first consumer: `extra_input_cols:
[close]`, log-diffed internally into a realized log-return series before forecasting.

This deliberately did **not** go the `model(x, **extras)` / separate-tensor route
sketched below — see the `x`-packing design discussion this shipped from: keeping
`Trainer.train_epoch` / `validate` / `predict` + `run_infer` blind to any per-model extra
argument (they only ever do `x = batch["x"]; model(x)`) was judged more important than a
dedicated tensor, and one packed `x` was sufficient once the column group is
independently selectable and self-describing (`extra_input_dim`).

**Still open** (real gaps, not addressed by the above):

- **Missing-value handling.** `extra_input_cols` still goes through the same
  `nan_to_num(0.0)` as feature/cov — fine for z-scored features, wrong for a raw price
  series NaN would silently become a nonsense price. Needs its own policy (drop the
  sample, or forward/back-fill within-symbol) before a real (non-mock) raw price column
  with gaps is used.
- **Independent window length.** `extra_input_cols` shares `seq_len` with feature/cov —
  a model wanting more raw history than the from-scratch models' `L` (Kronos, longer
  Chronos contexts) has no way to ask for it independently. Not needed yet; revisit if a
  model actually wants it.
- **OHLCV / multi-field raw schema (Kronos).** `extra_input_cols` is just "N more numeric
  columns packed into x" — fine for a single raw series like `close`, but Kronos wants a
  structured OHLCV schema, which is a different shape of "extra", not just more columns.

## C7 — backtester / metrics hardcode column names

`SimpleLongShortBacktester` hardcodes `pred_prob` and `fwd_logret_1`. Multi-horizon
targets (`fwd_logret_3` / `_6`) need a configurable return column. Tie this to C1's
`score` rename.

## C8 — model-supplied optimizer construction order

If a model gains `configure_optimizers(cfg) -> Optimizer | None` (for LoRA param groups /
discriminative LR — see `pretrained_research.md` §8), `Trainer.__init__` must resolve the
optimizer (model hook else default AdamW) **before** building the scheduler, which
references `self.optimizer`.

## C9 — artifact schema / wandb key migration

Changing `pred_prob -> score` (C1) touches `test_predictions.csv`, `predictions.csv`,
`metrics.json` keys, and wandb summary/log keys — anything downstream that reads those
files needs updating in lockstep. Also: `train/loss` is now a per-model quantity, so
cross-model loss curves in wandb are no longer comparable (leave as-is, just be aware).

## C10 — multi-target routing in the dataset — Done

`data.target_col` (singular) is now `data.target_cols` (a list) → `batch["y"]` is
`[B, target_dim]`, `target_dim` auto-injected on `BaseClassifierModel` alongside
`feature_dim`/`cov_dim`/`extra_input_dim`. A model that needs a target column beyond the
default single one just lists it in `target_cols`; no dataset/collate change needed per
model. `fwd_ret_col` (the Rank-IC ground truth, `batch["fwd_logret"]`) stays deliberately
**separate** from `target_cols` — singular, mandatory, shared by every model in a run —
so cross-model Rank IC comparability in a benchmark can't be broken by one model quietly
redefining "the forward return" for itself.
