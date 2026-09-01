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

Point-forecast / OHLCV models need a raw per-symbol series window `[L, k]` (e.g. raw
`close` / `logret`, or OHLCV for Kronos), which is **not** the single `fwd_logret`
scalar the dataset carries today.

**Fix:**
- `PanelTimeSeriesDataset`: opt-in `raw_cols` → precompute a separate array, surface an
  extra `[L, k]` tensor per sample. **Different missing-value handling** — cannot
  `nan_to_num(0.0)` a raw price series; drop the sample or normalize per the model.
- `panel_collate_fn`: stack the extra key.
- `Trainer.train_epoch` / `validate` / `predict` + `run_infer` loop: pass batch extras
  through as `model(x, **extras)` (the `forward(x, **kwargs)` hook already exists).

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

## C10 — multi-target routing in the dataset (only if needed)

Today a benchmark run shares one `target_col`. A model's `compute_loss` can already read
`batch["fwd_logret"]` instead of `batch["y"]`, so mixed binary + regression models in one
run mostly work without dataset changes. If a task needs a target column that is neither
`target_col` nor `fwd_ret_col` (e.g. `logret3_win`), add `extra_target_cols` passthrough
to `PanelTimeSeriesDataset` / `panel_collate_fn`.
