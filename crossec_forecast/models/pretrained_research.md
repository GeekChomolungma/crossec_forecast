# Pretrained Time-Series Foundation Models — Integration Research

> Scope: survey of candidate TSFMs to wrap as `PretrainedBackboneModel` subclasses and
> register into `crossec_forecast/models/registry.py`. Focus is on **how each library is
> actually used from Python**, **what inputs it demands beyond a plain `x` tensor**, and
> **how it can be mapped onto this repo's cross-sectional classification contract**.
>
> Last web pass: 2026-08. Version / date details below move fast — re-check before wiring.

---

## 0. The core mismatch to keep in mind

This framework is a **cross-sectional binary classifier**:

- input per sample: `x ∈ [B, L=6, D≈24]` — MAD-zscored *cross-sectional* features, not raw prices.
- label: `logret1_win ∈ {0,1}` (beat cross-sectional median next bar).
- selection / eval: **cross-sectional Rank IC** + top-bottom spread + long-short backtest.
- model contract: `forward(x, **kwargs) -> logits [B, 1]`.

Every model below is natively a **forecaster** (predict future values of a series) or a
**representation model** (masked reconstruction). None of them emit a "cross-sectional
score" directly. So a wrapper has to bridge one of three ways:

| Pattern | What the wrapper does | Fits current data layer? | Notes |
| --- | --- | --- | --- |
| **A. embedding + trainable head** | run the backbone's encoder on the `[L, D]` window → pooled embedding → small `nn.Linear(-> 1)` head (the head is the only thing trained; backbone frozen or LoRA) | **Yes**, works with `x=[B,6,24]` as-is | best fit for `PretrainedBackboneModel`'s "small trainable head" design. `L=6` is very short for these models but embeddings tolerate it better than autoregressive rollout. Needs the model to expose hidden states / an `encode()` path. |
| **B. forecast → signal** | feed a per-symbol raw target series (e.g. `close` or `logret`), get a point forecast of next-bar return, use forecast (or its z-score) as the logit; optionally pass the 24 features as **covariates** | **No** — dataset gives zscored features, not raw price; `L=6` far below these models' useful context (512–16k) | requires data-layer work (surface raw series + longer lookback). This is the "honest" use of a forecaster and the one where **covariate inputs matter**. |
| **C. zero-shot baseline (no training)** | pure zero-shot forecast of forward return as an alpha factor, piped straight into `eval/` + `backtest.py`, no `Trainer.fit` | N/A (bypasses trainer) | cheap reference row for the benchmark table; doesn't exercise the plugin/training path. |

**Implication for the `model(x)` vs `model(x, **kwargs)` decision:** only Pattern B needs
extra tensors threaded through the loader/Trainer. Under Pattern A everything the backbone
needs is already inside `x`. See the requirements matrix in §4.

### A vs B is not "packed into x vs passed separately" — it's *which part of the model you use*

A common misread: "B just adds an auxiliary series alongside `x`, so B is a strict
superset of A — why bother with A?". That's wrong. The split is about **what the pretrained
model computes and what objective trains**:

> **The root distinction: A uses the backbone's *encoder as a representation* and bolts on
> *our own trainable head*; B uses the backbone's *own native forecast head*.**
>
> The visible symptom is the output form — A emits a probability(-like) score, B emits a
> point forecast (a return value, a physical quantity) — but that symptom follows from
> which part of the model you kept. Everything else in this section (loss, target,
> whether the 24 features are used, comparability) is downstream of that one choice.

| | **Pattern A — representation** | **Pattern B — forecast** |
| --- | --- | --- |
| What the backbone outputs | an embedding vector for the window | a numeric forecast of the next return (maybe with quantiles) |
| What gets trained, on what loss | *our* small head, BCE on `logret1_win`, model-selected on val Rank IC | the model's own forecast head, on a **forecasting loss** (MSE/quantile) vs realized returns — or nothing at all (zero-shot) |
| Reuses the current `Trainer` / loss / early-stop as-is? | **Yes**, untouched | **No** — different target & loss, or bypasses `Trainer` (that's Pattern C) |
| The 24 engineered features | *are* the input, fed straight through | only used if the model takes covariates (Chronos-2 / TimesFM-3 / Moirai / TTM); the univariate forecasters **ignore them entirely** |
| Output → `[B,1]` logit | head emits it directly; AUC/F1/BCE all still meaningful | must convert a return forecast into a score; BCE/AUC/accuracy don't directly apply (works for Rank IC / long-short only) |

Why A stays the default and B does **not** subsume it:

1. **Fair benchmark.** The whole point is comparing models *on the same task* — same label
   (`logret1_win`), same loss, same Rank-IC selection, same split. Pattern A keeps every
   TSFM on exactly that task next to MLP/LSTM/DLinear. Pattern B is a *different experiment*
   ("can a zero-shot / forecasting-fine-tuned return prediction act as an alpha factor"),
   so its numbers aren't apples-to-apples with the from-scratch classifiers.
2. **B throws away feature engineering for most models.** Only 4 of the surveyed models
   consume covariates. For Chronos-T5 / Timer / Time-MoE / TiRex / Sundial, Pattern B
   forecasts one raw series and never sees the 24 `crossec_*` features — the exact signal
   `cross_section_enrich` was built to produce.
3. **B needs raw non-stationary series the panel deliberately hides.** You'd re-surface raw
   `close`/`logret` per symbol, choose a lookback, and match each model's expected scaling —
   losing the cross-sectional normalization that makes the features comparable.
4. **1-bar-ahead crypto return is ~unforecastable as a point value.** As a *representation*
   (A) the backbone's temporal features can still rank names cross-sectionally even when
   absolute forecasting is hopeless; as a *forecast* (B) you mostly get persistence /
   mean-reversion.
5. **"A needs extra parsing of `x`" is backwards.** In A the wrapper just calls the
   backbone's encoder on `x=[B,L,D]` as-is — the "packing" already happened in the data
   pipeline. **B is the one that adds work**: new dataset fields, collate changes, 3 Trainer
   call-site changes, a new loss/target, and forecast→score conversion.

Bottom line: do **A for all** models (fair, no core changes), and run **B as a separate
track for the ~4 covariate-aware models** where "24 features as exogenous inputs to a
forecaster" is a genuinely interesting second question — not as a replacement for A.

---

## 1. Environments — TSFM backends split by Python version

TSFM libraries pull mutually-incompatible dependency pins and Python-version constraints,
so several **cannot coexist in one interpreter** (e.g. `chronos-forecasting` runs on a
recent Python; `momentfm`'s older `transformers` pin wants Python ≤3.11). Backends
therefore live in **separate environments, classified by Python version — never by env
name** (env names are yours to choose):

| backend needs | extra | example models |
| --- | --- | --- |
| Python 3.11+ | `pip install -e ".[chronos]"` | `chronos_bolt`, `chronos2`, ... |
| Python 3.9–3.11 | `pip install -e ".[moment]"` | `moment` |

(Confirm each package's exact supported range from its own metadata.)

**How one codebase serves every environment (implemented):**

- A wrapper declares `REQUIRED_MODULES = ("momentfm",)` and an optional
  `PYTHON_HINT = "Python 3.9-3.11"`, and does its heavy `import` **inside `__init__`** via
  `models._optional.require_modules(...)` — never at module top level. So
  `import crossec_forecast` works in every environment.
- The wrapper class **still registers** where its backend is absent (shows in
  `list_registered_models()`, nameable in a config) but `build_model()` raises
  `ModelDependencyError` (an `ImportError` subclass) with an actionable message — install
  the extra, run from an interpreter matching `PYTHON_HINT` — instead of a bare crash.
- `is_model_available(name)` / `list_available_models()` report what is runnable in the
  current interpreter.
- **`BenchmarkEngine` scans the whole roster and silently skips (with a warning) any model
  whose backend is missing here**, so *one shared `benchmark.models` list* runs its
  available subset in every environment — no per-env config editing. Single-model
  `scripts/train.py` / `infer.py` instead fail loud (you named that model explicitly).

**Workflow:** edit any wrapper from your primary env. Run each model from an interpreter
that has its backend (`<py311+>/python scripts/train.py …`, `<py310>/python …`) — same
configs, same scripts. To benchmark everything, run `scripts/benchmark.py` once per env
against the same `experiment.yaml`; each writes a `benchmark_summary.csv` sorted by
`test_rank_ic`, so `concat` + re-sort gives the unified table.

---

## 2. Summary table

One table: the models you named first, then recent / trending ones (2025–2026) continuing
in the same columns.

| Model | Org | Repo | Params (open) | License (weights) | Native task | Max context | Covariates / exogenous | Best-fit pattern here | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TimesFM** | Google Research | `google-research/timesfm` | 200M (2.5), 500M (2.0), 3.0 out | Apache-2.0 (≤2.5); **3.0 non-commercial** | point/quantile forecast, decoder-only | 2k (2.0) → 16k (2.5) | 2.0: xreg API; **3.0: past-only + past&future covariates, native multivariate** | A or B | LoRA fine-tune (2.5+); don't ship 3.0 weights commercially |
| **Moirai / Moirai-2** | Salesforce AI | `SalesforceAIResearch/uni2ts` | S/B/L 14M–311M (1.x); 2.0-R-small ~11M | **CC-BY-NC-4.0** (non-commercial) | masked-encoder (1.x) / decoder (2.0), prob. forecast | patch-based, long | **Yes** — dynamic real covariates via `feat_dynamic_real` (GluonTS-style) | A or B | 2.0 = decoder rewrite, high GIFT-Eval rank among non-leaking models |
| **Chronos / Chronos-Bolt** | Amazon Science | `amazon-science/chronos-forecasting` | 8M–710M (T5), 9M–205M (Bolt) | Apache-2.0 | univariate forecast (tokenized / patched) | 512 (Bolt) | No (univariate only) | A (encoder embed) or C | Bolt ~250× faster / 20× less memory than original |
| **Chronos-2** | Amazon Science | same repo (`Chronos2Pipeline`) | 120M, encoder-only | Apache-2.0 | **univariate + multivariate + covariate-informed** zero-shot | group/patch based | **Yes, first-class** — `future_df` + extra columns | **B** (covariate-informed) or A | #1 on GIFT-Eval pretrained (2026-04); strongest match for this repo |
| **Kronos** | Shiyu Chen et al. (AAAI'26) | `shiyu-coder/Kronos` | mini 4.1M / small 24.7M / base 102M / large 499M (not open) | MIT | **financial K-line (OHLCV) autoregressive forecast** | 2048 (mini) / 512 (small,base) | Fixed schema: OHLC(+volume,+amount)+timestamps; no arbitrary covariates | **B** (finance-native) | only finance-pretrained option; forecast-only, no embedding head exposed |
| **Timer** | THU (thuml) | `thuml/Large-Time-Series-Model` | 84M (`thuml/timer-base-84m`) | MIT | decoder-only, forecast / imputation / anomaly | long (S3 flattening) | Univariate (Timer-XL adds multivariate) | A or B | HF API, `trust_remote_code=True` |
| **Timer-XL** | THU (thuml) | `thuml/Timer-XL` | ~ (checkpoints on HF) | MIT | unified multivariate forecast, long-context | long | Explicit multi-dim series; no dedicated covariate API | A or B | ICLR'25 |
| **Sundial** | THU (thuml) | same as Timer repo | 128M (`thuml/sundial-base-128m`) | MIT | **generative** (flow-matching) prob. forecast | variable | Univariate | A or B | ICML'25 oral; ms-latency |
| **UniTS** | Harvard MIMS (Zitnik lab) | `mims-harvard/UniTS` | small (checkpoints in Releases) | MIT | **multi-task**: forecast + classification + imputation + anomaly, prompt/task tokens | moderate | Multivariate; no covariate API | **A** (has a classification head already) | least packaged; needs vendoring the model code |
| **MOMENT** | CMU Auton Lab | `moment-timeseries-foundation-model/moment` | small/base/large (`AutonLab/MOMENT-1-*`) | MIT | masked reconstruction → embed / forecast / classify / anomaly / impute | **fixed 512** patches | Multi-channel; no covariate API | **A** (`task_name="embedding"` or `"classification"`) | `pip install momentfm`; pad the 6-step window to 512 |
| **Toto 2.0** | Datadog | `DataDog/toto` | 4M – 2.5B (u-μP scaled family) | Apache-2.0 (open weights) | multivariate prob. forecast, time/variate alternating attention | long | Multichannel; no explicit covariate API | A or B | trained on 2T+ points; observability-tuned (BOOM) — caution for finance |
| **Time-MoE** | Time-MoE team (ICLR'25 spotlight) | `Time-MoE/Time-MoE` | up to 2.4B sparse MoE (`Maple728/TimeMoE-{50M,200M}`) | Apache-2.0 | decoder-only MoE forecast, arbitrary horizon | 4096 | Univariate | A or B | cheap inference despite size (sparse activation) |
| **TiRex** | NX-AI (xLSTM group) | `NX-AI/tirex` | 35M | NXAI community license (check commercial terms) | xLSTM zero-shot forecast, quantile output | — | Univariate | A or B | strong on GIFT-Eval + Chronos-ZS; **Linux + NVIDIA CC≥8.0 only** |
| **IBM Granite TinyTimeMixer (TTM)** | IBM Research | `ibm-granite/granite-timeseries-ttm-r2` / r3 | **1M–5M** ("tiny") | Apache-2.0 | MLP-mixer multivariate forecast | short / fixed contexts | **First-class exogenous + static categorical infusion** | **B** (covariate-aware) | cheapest way to test "24 features as covariates" end-to-end; very fast |
| **TabPFN-TS** | Prior Labs | `PriorLabs/tabpfn-time-series` | small | check | tabular-PFN adapted to TS, zero-shot | — | tabular features | A or C | conceptually closest to a cross-sectional tabular model |
| **TimeGPT** | Nixtla | `Nixtla/nixtla` (API client) | closed (hosted API) | commercial API | forecast via hosted API | — | exogenous supported via API | **C only** | not open-weights — external baseline via API key only |

---

## 3. Per-model detail

### 3.1 TimesFM (Google Research)

- **Repo:** https://github.com/google-research/timesfm • checkpoints on HF (`google/timesfm-2.5-200m-pytorch`, `google/timesfm-3.0-pytorch`).
- **Architecture:** decoder-only, patched input, point + optional quantile head. PyTorch primary; JAX/Flax also available.
- **Versions:** 2.0 (500M, ctx 2048), 2.5 (200M, ctx 16k, optional 30M quantile head, horizon to 1k), 3.0 (native multivariate, past-only & past+future covariates, better zero-shot). **License trap:** source Apache-2.0, but **3.0 weights are non-commercial**; 2.5 and earlier weights Apache-2.0.
- **API (2.5-style):**
  ```python
  # forecaster.predict_batch(contexts, horizon,
  #     past_only_covariates=None, past_future_covariates=None,
  #     return_quantiles=True)
  ```
  2.0 had a separate `forecast_with_covariates(...)` path taking dynamic/static numerical & categorical regressors and doing an internal linear xreg fit.
- **Fine-tuning:** LoRA via HF Transformers + PEFT (examples in repo, 2.5+).
- **Fit here:** Pattern A — take patch embeddings / last hidden state over the `[6,24]` window (treat each of 24 channels as a covariate or stack), add a head. Pattern B — per-symbol return series as target + 24 features as `past_future_covariates` (3.0). Very short `L=6` is the main obstacle for B.

### 3.2 Moirai / Moirai-2 (Salesforce, `uni2ts`)

- **Repo:** https://github.com/SalesforceAIResearch/uni2ts • weights `Salesforce/moirai-1.0-R-{small,base,large}`, `Salesforce/moirai-2.0-R-small`.
- **Architecture:** 1.x = masked-**encoder** universal transformer, "any-variate" attention, multi-patch-size, distributional head. 2.0 = decoder, quantile loss, multi-token prediction, patch-level masking.
- **Context / covariates:** patch-based, handles long context; **exogenous supported** — targets + dynamic real covariates are packed jointly via `feat_dynamic_real_dim` and partitioned into patches (GluonTS `PandasDataset` / `feat_dynamic_real`). Known-future covariates supported.
- **API:** `from uni2ts.model.moirai import MoiraiForecast, MoiraiModule`; wraps into a GluonTS predictor. Fine-tune / pretrain via the uni2ts CLI (hydra configs).
- **License:** **CC-BY-NC-4.0 weights** — non-commercial only. Flag if this repo's output is ever productionized.
- **Fit here:** Pattern A — Moirai-1.x encoder is a natural embedding extractor for `[6,24]` (any-variate attention already ingests D channels). Pattern B — target=fwd return, 24 feats as `feat_dynamic_real`.

### 3.3 Chronos family (Amazon Science)

- **Repo:** https://github.com/amazon-science/chronos-forecasting • `pip install chronos-forecasting` • Apache-2.0 (source **and** weights).
- **Chronos-T5 (original):** 8M–710M, LLM-style — quantize series into tokens, T5 seq2seq, sample trajectories. Univariate.
- **Chronos-Bolt:** 9M–205M, patch-based direct multi-step, ~250× faster / 20× less memory than original. Univariate. Good cheap embedding backbone.
- **Chronos-2 (2025-10):** 120M **encoder-only**, single architecture covering **univariate + multivariate + covariate-informed** zero-shot; largest gains exactly on tasks with exogenous features; ranked #1 on GIFT-Eval pretrained board (2026-04).
  ```python
  from chronos import Chronos2Pipeline
  pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
  pred_df = pipe.predict_df(context_df, future_df=future_df, prediction_length=24,
                            quantile_levels=[0.1,0.5,0.9],
                            id_column="id", timestamp_column="timestamp", target="target")
  ```
  Covariates = extra columns in `context_df` + their known-future values in `future_df`.
- **Fit here:** **Chronos-2 is the strongest match** for this repo's use case (24 exogenous cross-sectional features → forecast forward return) — Pattern B, covariate-informed. Also usable as Pattern A (encoder embeddings). Older Chronos = Pattern A / C only.

### 3.4 Kronos (financial K-line foundation model)

- **Repo:** https://github.com/shiyu-coder/Kronos • weights `NeoQuasar/Kronos-{mini,small,base}` + tokenizers `NeoQuasar/Kronos-Tokenizer-{base,2k}` • MIT • AAAI 2026.
- **Design:** two-stage — a learned tokenizer quantizes multi-dim **OHLCV** bars into hierarchical discrete tokens; a decoder-only autoregressive transformer over those tokens. Purpose-built for the "language of markets", trained on 45+ exchanges.
- **Variants:** mini 4.1M (ctx 2048) / small 24.7M (ctx 512) / base 102M (ctx 512) / large 499M (weights not released).
- **API:**
  ```python
  from model import Kronos, KronosTokenizer, KronosPredictor
  tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
  mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
  predictor = KronosPredictor(mdl, tok, device="cuda", max_context=512)
  pred_df = predictor.predict(df=hist_ohlcv,           # cols: open,high,low,close[,volume,amount]
                              x_timestamp=hist_ts, y_timestamp=future_ts,
                              pred_len=H, T=1.0, top_p=0.9, sample_count=1)
  ```
  Outputs a DataFrame of forecast OHLCV. **Forecast only — no embedding/feature head exposed** (would need to hook internal hidden states manually). `predict_batch()` for many symbols.
- **Input demand:** a specific schema (OHLCV + real timestamps), *not* arbitrary covariates. This is a distinct kind of "extra input" — the wrapper needs raw bars, which this repo's dataset does not currently pass.
- **Fine-tuning:** supported (`finetune/train_tokenizer.py`, `finetune/train_predictor.py`, `torchrun`, Qlib for data prep; A-share example).
- **Fit here:** Pattern B, finance-native — feed each symbol's OHLCV window, use forecast next-bar logret as the score. Needs data-layer change to surface raw OHLCV. Pattern A only if we fork the model to return hidden states.

### 3.5 Timer / Timer-XL / Sundial (Tsinghua `thuml`)

- **Repos:** https://github.com/thuml/Large-Time-Series-Model (Timer + Sundial), https://github.com/thuml/Timer-XL, fine-tune tooling in https://github.com/thuml/OpenLTM • MIT.
- **Timer** (ICML'24): decoder-only GPT-style, "S3" (single-series sequence) flattening; tasks = forecast / imputation / anomaly. `thuml/timer-base-84m`.
- **Timer-XL** (ICLR'25): long-context, explicit multi-dimensional modeling, unified forecasting; for supervised or large-scale pretraining.
- **Sundial** (ICML'25 oral): generative (flow-matching / "TimeFlow" head), trained on ~1T points, point + probabilistic, ms-latency. `thuml/sundial-base-128m`.
- **API (HF, uniform):**
  ```python
  from transformers import AutoModelForCausalLM
  m = AutoModelForCausalLM.from_pretrained("thuml/timer-base-84m", trust_remote_code=True)
  out = m.generate(seqs, max_new_tokens=H)     # seqs: [B, lookback], univariate
  ```
- **Covariates:** none native (Timer/Sundial univariate; Timer-XL takes multi-dim but no dedicated exogenous channel).
- **Fit here:** Pattern A — grab hidden states from the causal stack over the flattened window; per-channel or concat, then head. Pattern B for a single target series. `trust_remote_code=True` required.

### 3.6 UniTS (Harvard MIMS / Zitnik lab)

- **Repo:** https://github.com/mims-harvard/UniTS • pretrained weights in GitHub Releases • MIT • NeurIPS 2024.
- **Design:** one backbone (sequence + variable attention, dynamic linear operator) for **forecasting, classification, imputation, anomaly** with **no task-specific heads** — tasks selected via task tokenization + learnable prompt tokens. Strong zero-/few-shot + prompt learning across 38 datasets incl. finance.
- **Covariates:** multivariate input; no separate exogenous API.
- **Maturity:** small repo, low commit count, no packaged `pip` inference API — expect to vendor the model code and its config. Tutorial.md covers custom data.
- **Fit here:** **Pattern A with the least glue** — UniTS already has a classification task mode producing class logits; point it at the `[6,24]` window with `num_class=2` (or forecasting mode + head). The prompt-token mechanism could even absorb a "cross-sectional rank" framing.

### 3.7 MOMENT (CMU Auton Lab)

- **Repo:** https://github.com/moment-timeseries-foundation-model/moment • `pip install momentfm` • weights `AutonLab/MOMENT-1-{small,base,large}` • MIT • ICML 2024.
- **Design:** T5-encoder over fixed **512**-length patched input, pretrained by masked patch reconstruction. One model, many task heads.
- **API:**
  ```python
  from momentfm import MOMENTPipeline
  model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
             model_kwargs={"task_name": "embedding"})            # or "classification"/"forecasting"/"reconstruction"
  # classification: model_kwargs={"task_name":"classification","n_channels":24,"num_class":2}
  ```
- **Input:** multi-channel `[B, n_channels, 512]`; short series are padded/masked to 512. No covariate concept (all channels are "the series").
- **Covariates:** none — but `n_channels=24` naturally ingests all cross-sectional features as channels.
- **Fine-tuning:** PEFT demonstrated (ECG tutorial); full training via `moment-research`.
- **Fit here:** **Pattern A, clean** — `task_name="embedding"` → `[B, d]` embedding of the `[24, 6→pad512]` window → trainable head; or `task_name="classification"` with `num_class=2` and let MOMENT's own head be the trainable part. Padding 6→512 is wasteful but works; consider `MOMENT-1-small`.

### 3.8 Extras (trending)

- **Toto 2.0 (Datadog)** — https://github.com/DataDog/toto, weights `Datadog/Toto-Open-Base-1.0` and the 2.0 family (4M–2.5B). Multivariate, alternating time/variate attention, quantile output, 2T+ training points. Apache-2.0 open weights. Pattern A (variate attention ingests the 24 channels) or B. Their **BOOM** benchmark is observability-flavored, not finance — treat leaderboard numbers with caution for our domain.
- **Time-MoE** — https://github.com/Time-MoE/Time-MoE, `Maple728/TimeMoE-{50M,200M}` (+ up to 2.4B). Decoder-only sparse MoE, ctx 4096, arbitrary horizon, Apache-2.0. Pattern A/B; cheap at inference despite size.
- **TiRex (NX-AI)** — https://github.com/NX-AI/tirex, `NX-AI/TiRex`, 35M xLSTM, quantile forecasts, top GIFT-Eval / Chronos-ZS. **Runs only on Linux + NVIDIA GPU CC≥8.0**; license is an NXAI community license (verify commercial use). Pattern A/B; attractive because tiny + strong.
- **IBM Granite TinyTimeMixer (TTM)** — `ibm-granite/granite-timeseries-ttm-r2` / `-r3`, **1–5M params**, Apache-2.0, MLP-mixer. **First-class exogenous + static-categorical infusion** and it's fast. Best cheap covariate-aware Pattern-B baseline; also a sanity check that a non-attention TSFM can use our 24 features as exogenous channels.
- **TabPFN-TS (Prior Labs)** — tabular PFN adapted to time series; conceptually the closest to a cross-sectional tabular learner, strong zero-shot GIFT-Eval. Worth a row if licensing/API allow.
- **TimeGPT (Nixtla)** — closed, hosted API only (`nixtla` client + API key). Only viable as a Pattern-C external baseline, not a wrapped plugin.

---

## 4. Extra-input requirement matrix (drives the `model(x)` vs `model(x, **kwargs)` decision)

**How to read this table.** Each row is a candidate model. Columns 2 and 3 answer: *"if we
wrap it with that pattern, does the wrapper need any tensor beyond the standard
`x = [B, 6, 24]`?"* — i.e. would we have to thread an extra input through
`dataset → collate → Trainer → model(x, **kwargs)`. "No" means `forward(x)` alone is
enough. Column 5 is the pattern we'd actually pick for that model, and why.

Pattern legend: **A** = run the backbone as an encoder on the `[6,24]` window → pooled
embedding → small trainable head. **B** = feed a raw per-symbol series, take the model's
forecast of forward return as the logit (features optionally passed as covariates).
**C** = zero-shot forecast used directly as an alpha factor, no `Trainer.fit`.

| Model | Pattern A — extra input beyond `x`? | Pattern B — extra input beyond `x`? | Extra input Pattern B needs | **Recommended pattern + why** |
| --- | --- | --- | --- | --- |
| TimesFM 2.5 | No | Yes | raw target series (long); covariates optional | **A** — `L=6` makes autoregressive rollout pointless; embed the window + head |
| TimesFM 3.0 | No | Yes | raw target + optional past/future covariates (the 24 feats) | **B if covariates threaded**, else A — 3.0's covariate path is the whole reason to pick it over 2.5; only worth it once raw returns + a covariate tensor exist |
| Moirai 1.x / 2 | No | Yes | target + `feat_dynamic_real` (24 feats), optional known-future | **A** — the any-variate encoder already ingests the 24 channels directly; B later. Non-commercial weights |
| Chronos-T5 / Bolt | No | Yes | raw target series only (univariate) | **A** (encoder embedding) or **C** — no covariate support, so B plumbing buys nothing |
| **Chronos-2** | No | **Yes** | `context_df` + `future_df` with target **and 24 covariate columns** | **B** — flagship covariate-informed use, the 24 feats map straight to covariate columns; also cheap **C** zero-shot. Worth threading covariates for this one alone |
| Kronos | No (needs a model fork to expose hidden states) | Yes | **raw OHLCV bars + real timestamps** per symbol (fixed schema) | **B** — it consumes *only* OHLCV; stuffing OHLCV into the zscored `x` channels would wreck `x`'s cross-sectional structure and defeat the point. Needs a dedicated raw-bars passthrough, not a generic covariate tensor |
| Timer / Timer-XL / Sundial | No | Yes | raw target series (univariate; Timer-XL multi-dim) | **A** — hidden states over the window + head; B adds nothing without covariates |
| UniTS | No | No / optional | multivariate window is enough; task token selects mode | **A** — it already ships a classification head: feed `[6,24]` with `num_class=2`, zero extra input |
| MOMENT | No | No | multichannel window padded to 512; no covariates | **A** — `task_name="embedding"` / `"classification"`, `n_channels=24`; only cost is padding 6→512 |
| Toto 2.0 | No | optional | multichannel; no explicit covariate API | **A** — variate attention ingests the 24 channels; B has no covariate hook |
| Time-MoE | No | Yes | raw target series | **A** — univariate forecaster; B needs a raw series and gives no covariate benefit |
| TiRex | No | Yes | raw target series | **A** (if Linux/GPU constraint is OK) — univariate, same reasoning as Time-MoE |
| IBM TTM | No | **Yes** | target + exogenous channels (24 feats) + static categoricals | **B** — native exogenous infusion *is* our 24-feature case, and at 1–5M params it's cheap to run the full B pipeline as a proof of concept |

**Reading of the matrix:**
- If we commit to **Pattern A across the board**, `forward(x)` is enough — no Trainer/loader
  changes. `**kwargs` stays a latent hook.
- The models that *reward* extra structured input are exactly the **covariate-informed
  forecasters**: **Chronos-2, TimesFM 3.0, Moirai, IBM TTM**. For those, Pattern B with the
  24 features as covariates is the intended use and likely the strongest signal — but it
  needs: (a) raw forward-return target surfaced per `(symbol, t)`, (b) a longer lookback
  option, (c) a covariate tensor threaded `dataset → collate → Trainer → model(x, covx=...)`.
- **Kronos** is a special case: finance-native, wants raw OHLCV — different plumbing than a
  generic covariate tensor.

Recommendation: finish wrapping the **Pattern A** versions first (no core changes), collect
how many of the shortlisted models we actually want in Pattern B, then do **one** generic
`batch extras -> forward(**kwargs)` change in `dataloader.py` + `trainer.py` (3 call sites)
rather than per-model hacks.

---

## 5. Suggested priority for this repo

1. **Chronos-2** — Apache-2.0, covariate-informed, current GIFT-Eval #1, `pip`-installable, clean `predict_df`. Do it both as Pattern A (embed+head) and Pattern C (zero-shot factor) first; Pattern B once covariates are threaded.
2. **MOMENT (small/base)** — MIT, `momentfm` packaged, `task_name` switch makes Pattern A almost trivial (`n_channels=24`, `num_class=2`).
3. **IBM Granite TTM (r2/r3)** — Apache-2.0, 1–5M params, native exogenous infusion; the cheapest way to test the "24 features as covariates" hypothesis end-to-end.
4. **Kronos (small/base)** — MIT, the only *finance-pretrained* option; high narrative value in the benchmark even if it needs OHLCV plumbing. AAAI'26.
5. **TimesFM 2.5** — Apache-2.0 (avoid 3.0 weights for commercial), strong general baseline, LoRA fine-tune path.
6. **Timer / Sundial** — MIT, uniform HF API, good for a "generic decoder-only TSFM" data point.
7. **Toto 2.0** — Apache-2.0, multivariate, size range lets us probe scaling; caveat: observability-tuned.
8. **Moirai-2** — strong, but **CC-BY-NC-4.0** weights; keep as research-only row.
9. **UniTS** — most conceptually aligned (built-in classification + prompt tokens) but least packaged; needs vendoring. Medium-term.
10. **TiRex** — tiny + strong, but Linux/NVIDIA-only and license needs checking; opportunistic.

Deferred / external-only: **TimeGPT** (closed API), **TabPFN-TS** (evaluate licensing).

---

## 6. Open questions before implementation

- **Lookback (decided: pin one `L` for everyone, target `L=512`).** `L=6` wastes the TSFMs
  and makes cross-model comparison unfair. `L=512` saturates the 512-context models
  (Chronos-2, Chronos-Bolt, Kronos-small/base) and removes MOMENT's 6→512 padding. It is a
  **config-only** change (`data.seq_len: 512`; `seq_len` is auto-injected into every model).
  MLP/LSTM/DLinear all still train fine at 512 (MLP first layer becomes `Linear(512*24, h)`;
  LSTM per-forward cost ~85× vs L=6 — plan sweep budget). No leakage is introduced: long
  windows reach back into train-era *feature* rows (past features → future label, legit); the
  1-bar horizon means `embargo_steps=1` still suffices regardless of `L`.
  **Done:** `data.seq_len: 512` set in `experiments/experiment.yaml` (YAML overrides the
  schema default of 6; `seq_len` then flows through `to_data_config → build_dataloaders →
  meta_info → model_build_config` into every model). Sample count confirmed ~3M+, so the
  `< seq_len` symbol drop and the lost leading `seq_len-1` windows are not a concern.
  Non-blocking follow-ups: bump the stale `run.tags` `L512` label; raise `data.num_workers`
  (per-sample gather is ~85× the bytes of L=6); budget LSTM wall-clock (~85× per forward);
  consider more regularization for the now-12288-wide MLP input layer. A later
  "max-context study" can still let long-context models use 4k+, but the head-to-head
  benchmark pins a single `L`.
- **Raw series access:** Pattern B / Kronos need `close` / `logret` / OHLCV per `(symbol, t)`.
  `PanelTimeSeriesDataset` currently exposes only zscored features + `fwd_logret` + meta.
  Add an opt-in `raw_cols` passthrough?
- **Frozen backbone + optimizer / checkpoints:** needs real (small) changes to `Trainer` and
  the wrappers — see the checklist in §7.
- **Dependency isolation:** each wrapper's heavy import (`chronos`, `momentfm`, `uni2ts`,
  `timesfm`, …) must be lazy (inside `__init__`) or `try/except`-guarded so `import
  crossec_forecast` doesn't hard-require all of them. Add `pyproject.toml` optional extras.
- **License propagation:** TimesFM 3.0 (non-commercial) and Moirai (CC-BY-NC-4.0) weights
  can't go into a commercial pipeline — tag these plugins clearly.

---

## 7. Follow-up engineering tasks (backbone freezing, LoRA, optimizer, checkpoints)

### Background: what `AdamW(self.model.parameters())` does today

[`Trainer.__init__`](../engine/trainer.py) builds one optimizer:

```python
self.optimizer = optimizer or torch.optim.AdamW(
    self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay,
)
```

- `AdamW` = Adam + decoupled weight decay. `self.model.parameters()` = **every** parameter
  tensor of the whole model (backbone + head), in **one group**, all on the same `lr` and
  `weight_decay`.
- The list is snapshotted **once**, at `Trainer` construction. `build_model(...)` fully
  builds the model before `Trainer(model=...)`, so a wrapper that freezes / injects LoRA in
  its own `__init__` is seen correctly; anything done lazily *after* that is missed.
- `Trainer` already *accepts* `optimizer=` / `scheduler=` kwargs, but the pipelines
  (`run_train`, `BenchmarkEngine`) never pass them — so today there is no way to get param
  groups without changing the pipeline or the model contract.

### Conflicts / limitations for a "freeze backbone + LoRA head" setup

| Concern | Current behaviour | Impact |
| --- | --- | --- |
| Freezing a subset | params with `requires_grad=False` produce no grad; PyTorch AdamW skips no-grad params entirely (no update, no weight decay) | **Correctness OK** — frozen stays frozen. Only sloppiness: passing them in at all. |
| Discriminative / layer-wise LR (backbone-LoRA small lr, head larger lr) | impossible — single group, single `lr` | standard fine-tuning recipe not expressible |
| `weight_decay=0` on LoRA / norm / bias params | impossible — one `weight_decay` for all | mild quality hit; PEFT convention violated |
| AMP (bf16), grad accumulation, grad checkpointing | none | 100M–500M backbones slower / heavier than they need to be |
| Checkpoint = full `state_dict` | `torch.save(model.state_dict())` in `fit()` | `_best.pt` bloats by the whole frozen backbone; `run_infer` does `weights_only=True` + strict load |

### Task checklist

- [ ] **T1 — trainable-param selection.** In `Trainer`, build the default optimizer from
      `self.model.trainable_parameters()` when the model defines it, else
      `self.model.parameters()`. Add a default `trainable_parameters()` on
      `BaseClassifierModel` returning `(p for p in self.parameters() if p.requires_grad)`.
      Backward-compatible (from-scratch models: all params require grad).
- [ ] **T2 — `configure_optimizers` hook.** Optional
      `BaseClassifierModel.configure_optimizers(cfg: TrainConfig) -> Optimizer | None`
      (default returns `None`). If a model returns an optimizer, `Trainer` uses it instead
      of the built-in AdamW. `PretrainedBackboneModel` subclasses use this to declare param
      groups: e.g. `[{params: lora_params, lr: cfg.lr*0.1, weight_decay: 0}, {params:
      head_params, lr: cfg.lr, weight_decay: cfg.weight_decay}]`. Nothing else changes.
- [ ] **T3 — wrapper LoRA plumbing.** Each `PretrainedBackboneModel` subclass `__init__`:
      load backbone from `self.pretrained_path` → `backbone.requires_grad_(not train)` per
      `self.freeze_backbone` → apply LoRA (`peft.get_peft_model`, or the backbone's native
      LoRA path e.g. TimesFM 2.5) → build the trainable head. Read
      `lora_rank / lora_alpha / lora_dropout / lora_target_modules` from `model.config`
      (the open `model.config` dict has no schema typo-protection — a misspelled key
      silently falls back to the default).
- [ ] **T4 — checkpoint slimming.** Wrapper overrides `state_dict()` to emit only trainable
      tensors (head + LoRA adapters); `load_state_dict(..., strict=False)` on reload, with
      the backbone rebuilt from `pretrained_path` in `__init__`. `run_infer` needs
      `strict=False` for these trimmed checkpoints (currently strict).
- [ ] **T5 — optimizer-timing invariant.** Document + assert: wrappers must finish all
      freezing / LoRA injection inside `__init__` (Trainer snapshots params once).
- [ ] **T6 — (later, only if needed) AMP + grad-accum.** Add `train.amp: none|bf16|fp16` and
      `train.grad_accum_steps` to `experiment_schema.py` + `Trainer`. Defer until a real
      backbone actually OOMs or is too slow.
- [ ] **T7 — pipeline check.** Confirm `run_train` / `BenchmarkEngine` still work when the
      model supplies its own optimizer via T2 (they pass `config` only — fine, T2 is
      model-side). Scheduler / early-stop / `min_delta` on val Rank IC are unaffected.

Note: T1+T2+T3+T4 are the minimum to fine-tune a pretrained backbone properly. T1 alone is
enough to *freeze* a backbone and train only a head at a single LR (Pattern A, no LoRA).

---

## Sources

- TimesFM — https://github.com/google-research/timesfm • https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/ • https://huggingface.co/google/timesfm-3.0-pytorch
- Moirai / uni2ts — https://github.com/SalesforceAIResearch/uni2ts • https://www.salesforce.com/blog/moirai/ • https://huggingface.co/Salesforce/moirai-2.0-R-small • https://huggingface.co/Salesforce/moirai-1.0-R-large
- Chronos / Chronos-2 — https://github.com/amazon-science/chronos-forecasting • https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting • https://huggingface.co/amazon/chronos-2 • https://arxiv.org/pdf/2510.15821
- Kronos — https://github.com/shiyu-coder/Kronos • https://huggingface.co/NeoQuasar/Kronos-base • https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k • https://shiyu-coder.github.io/Kronos-demo/ • https://explainx.ai/blog/kronos-foundation-model-financial-candlesticks-aaai-2026
- Timer / Timer-XL / Sundial — https://github.com/thuml/Large-Time-Series-Model • https://github.com/thuml/Timer-XL • https://huggingface.co/thuml/timer-base-84m • https://huggingface.co/thuml/sundial-base-128m • https://github.com/thuml/OpenLTM
- UniTS — https://github.com/mims-harvard/UniTS • https://openreview.net/forum?id=nBOdYBptWW • https://zitniklab.hms.harvard.edu/projects/UniTS/
- MOMENT — https://github.com/moment-timeseries-foundation-model/moment • https://huggingface.co/AutonLab/MOMENT-1-large • https://github.com/moment-timeseries-foundation-model/moment-research
- Toto — https://github.com/DataDog/toto • https://www.datadoghq.com/blog/ai/toto-boom-unleashed/ • https://huggingface.co/Datadog/Toto-Open-Base-1.0 • https://arxiv.org/pdf/2605.20119
- Time-MoE — https://github.com/Time-MoE/Time-MoE • https://arxiv.org/abs/2409.16040 • https://huggingface.co/Maple728/TimeMoE-200M
- TiRex — https://github.com/NX-AI/tirex • https://huggingface.co/NX-AI/TiRex • https://nx-ai.github.io/tirex/
- IBM Granite TTM — https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2 • https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3 • https://developer.ibm.com/tutorials/awb-foundation-model-time-series-forecasting/
- Leaderboard / landscape — https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/ • https://research.ibm.com/blog/SSM-time-series-model • https://arxiv.org/pdf/2605.20119
</content>
</invoke>
