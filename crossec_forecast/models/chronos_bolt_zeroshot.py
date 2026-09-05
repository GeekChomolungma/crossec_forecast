"""
Chronos-Bolt, zero-shot — Pattern B (native forecast head) x NO training: the frozen
backbone's own next-step forecast, read directly as the cross-sectional score.

Registered as ``chronos_bolt_zeroshot``. Unlike ``chronos_bolt_head_only`` there is no
trainable head and no per-channel-then-linear-probe compromise — this is Chronos-Bolt run
exactly the way it is meant to be used: feed it a univariate context window of realized
log-returns, read its own one-step forecast of the next log-return.

Recommended usage — point ``extra_input_cols`` (NOT ``feature_cols``) at a RAW,
strictly-positive price-like column:

  data:
    extra_input_cols: [close]      # <- raw price, packed into x after feature/cov

``extra_input_cols`` is a dedicated column-role (see ``PanelTimeSeriesDataset``'s
docstring) for "raw/passthrough columns a model needs for its own internal use" — kept
separate from ``feature_cols`` (whose established meaning elsewhere is "the crossec_*
z-scored panel") precisely so a wrapper like this one never has to overload that field.
``forward`` slices its own ``x[..., feature_dim+cov_dim : feature_dim+cov_dim+
extra_input_dim]`` and converts the raw price window to a log-return window internally
(``diff(log(price))``) before handing it to Chronos, so the model's own forecast comes out
in EXACTLY the same units and horizon as ``fwd_logret_1`` — ``log(close[t+1]) -
log(close[t])`` — not an approximation of it. This is what makes ``compute_loss``'s
default comparison against ``batch["fwd_logret"]`` correct *by construction*: an earlier
version of this wrapper pointed `feature_cols` at `crossec_logret_1_mad_Zscore` (a
*cross-sectionally z-scored* value) and still diffed it against the physical-scale
`fwd_logret_1` — same units label ("logret"), different scale, a real bug. Feeding a raw
price column through the dedicated `extra_input_cols` slot and log-diffing inside the
model is the fix.

Set ``model.config.input_is_price: false`` to skip the internal log-diff and feed a column
that is already a return-like series as-is (e.g. a future raw, non-z-scored `logret`
passthrough column) — accepting whatever scale/timing that column happens to carry.

  x[..., feature_dim+cov_dim : ...+extra_input_dim]  -- raw price window [B, L, D]
             --(input_is_price: diff(log(x)))--> realized log-return window [B, L(-1), D]
             --(per-channel, frozen Chronos-Bolt)--> 1-step forecast [B, D]
             --(mean across D)--> raw [B, 1]           (D==1 in the recommended usage -> no-op)
             --(to_score: * sign)--> score [B]          (higher == more bullish)

  output_kind = "point_forecast"
  zero_shot   = True   -> Trainer runs this eval-only (no optimizer, no training loop)
  compute_loss -> Huber/MSE of raw vs a config-selected batch field (`loss_target`, default
                  "fwd_logret" -> batch["fwd_logret"], i.e. fwd_ret_col; or "y" ->
                  batch["y"], i.e. target_cols, for a fragment that repoints `data.target_cols`
                  at something scale-matched instead). Never backpropped; val_loss only.

No `data.target_cols` / `data.feature_cols` override is needed for the default usage above:
`feature_cols` stays whatever the base config resolves (the crossec_* panel is simply
unused by this model — it only reads its own extra_input slice), and `target_cols` stays
the base `["logret1_win"]`, unused by the default `loss_target: fwd_logret`; the dataset
still requires *some* valid target_cols to not drop every sample (see the config fragment).

Backend: `chronos-forecasting` (import name `chronos`). Recent Python only
(`pip install -e ".[chronos]"`).

config keys (model.config):
  model_id             "amazon/chronos-bolt-{tiny,mini,small,base}"  (default: small)
  input_is_price        bool         default true: log-diff the input before forecasting
                        (see module docstring). false = feed the column through unchanged.
  context_length        int | null   null = use the full data.seq_len window; counts
                        RAW input steps (pre-diff), so the fed return series is one shorter
                        when input_is_price is true.
  prediction_length     int          default 1 (next bar)
  reduction             "mean"       how the D per-channel forecasts collapse to one raw
                        scalar (only "mean" is implemented; a no-op when D==1)
  sign                  +1.0 | -1.0  flip the factor if val Rank IC comes out negative
  loss_target           "fwd_logret" | "y"   which batch field compute_loss compares raw
                        against (default "fwd_logret" — see module docstring)
  loss_type             "huber" | "mse"   (default huber; diagnostic val_loss only)
  huber_delta           float        default 1.0
  series_chunk          int          max series per Chronos call (default 4096)
  torch_dtype           str | null   e.g. "bfloat16"; null = fp32
"""
from typing import Dict, Any

import torch
import torch.nn.functional as F

from .pretrained import PretrainedBackboneModel
from .registry import register_model
from ._optional import require_modules


@register_model("chronos_bolt_zeroshot")
class ChronosBoltZeroShot(PretrainedBackboneModel):
    REQUIRED_MODULES = ("chronos",)
    PYTHON_HINT = 'recent Python; pip install -e ".[chronos]"'
    output_kind = "point_forecast"
    zero_shot = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # sets self.pretrained_path, self.freeze_backbone
        chronos = require_modules(
            "chronos_bolt_zeroshot", self.REQUIRED_MODULES,
            extra="chronos", python_hint=self.PYTHON_HINT,
        )["chronos"]

        if self.extra_input_dim <= 0:
            raise ValueError(
                "chronos_bolt_zeroshot needs at least one extra_input column (a raw "
                "price-like series) — set data.extra_input_cols, e.g. [close]. It "
                "deliberately does NOT read feature_cols (see the module docstring)."
            )

        self.model_id = str(config.get("model_id") or self.pretrained_path or "amazon/chronos-bolt-small")
        self.input_is_price = bool(config.get("input_is_price", True))
        self.context_length = config.get("context_length")
        self.pred_length = int(config.get("prediction_length", 1))
        self.series_chunk = int(config.get("series_chunk", 4096))
        self._reduction = str(config.get("reduction", "mean")).lower()
        if self._reduction not in ("mean",):
            raise ValueError("chronos_bolt_zeroshot: reduction must be 'mean'")
        self._sign = float(config.get("sign", 1.0))
        self._loss_target = str(config.get("loss_target", "fwd_logret")).lower()
        if self._loss_target not in ("fwd_logret", "y"):
            raise ValueError("chronos_bolt_zeroshot: loss_target must be 'fwd_logret' or 'y'")
        self._loss_name = str(config.get("loss_type", "huber")).lower()
        self._huber_delta = float(config.get("huber_delta", 1.0))

        if not self.freeze_backbone:
            raise NotImplementedError(
                "chronos_bolt_zeroshot is a frozen zero-shot baseline; set freeze_backbone=True. "
                "To train a head on top of Chronos-Bolt use chronos_bolt_head_only instead."
            )

        from_kwargs: Dict[str, Any] = {}
        torch_dtype = config.get("torch_dtype")
        if torch_dtype:
            from_kwargs["torch_dtype"] = getattr(torch, str(torch_dtype))
        _pipe = chronos.BaseChronosPipeline.from_pretrained(self.model_id, **from_kwargs)
        for p in _pipe.model.parameters():
            p.requires_grad_(False)
        _pipe.model.eval()
        # Plain attribute, NOT registered as a submodule (mirrors chronos_bolt_head_only /
        # moment_zeroshot): keeps the frozen backbone out of parameters() / state_dict().
        object.__setattr__(self, "_pipe", _pipe)

        # This wrapper carries no parameters at all; the buffer gives `model.to(device)`
        # something to move and lets `_channel_forecasts` read the Trainer's target device
        # (same trick as moment_zeroshot's `_dev_anchor`).
        self.register_buffer("_dev_anchor", torch.zeros(1), persistent=True)

    # -- slice this model's own input out of the shared packed `x` --------------------
    def _extra_input_slice(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, feature_dim+cov_dim+extra_input_dim] -> this model's own
        [B, L, extra_input_dim] block. See PanelTimeSeriesDataset's packing convention."""
        start = self.feature_dim + self.cov_dim
        end = start + self.extra_input_dim
        return x[..., start:end]

    # -- frozen Chronos forecast per input channel -----------------------------------
    @torch.no_grad()
    def _channel_forecasts(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, D] -> [B, D] one-step-ahead forecast per channel (detached).

        When ``input_is_price`` (default), each channel is a raw price-like level series:
        it is log-differenced to a realized log-return series BEFORE being handed to
        Chronos, so the returned forecast is itself a next-step *return*, in the same
        units as `fwd_logret_1` — see the module docstring for why this (not a
        pre-z-scored feature column) is the correct native-usage input.
        """
        b, L, d = x.shape
        ctx = x.permute(0, 2, 1).reshape(b * d, L)  # [B*D, L] raw per-channel windows

        if self.context_length:
            take = int(self.context_length) + (1 if self.input_is_price else 0)
            ctx = ctx[:, -take:]

        if self.input_is_price:
            log_ctx = torch.log(ctx.clamp_min(1e-8))
            ctx = log_ctx[:, 1:] - log_ctx[:, :-1]  # [B*D, L(-1)] realized log-returns

        dev = self._dev_anchor.device
        if next(self._pipe.model.parameters()).device != dev:
            self._pipe.model.to(dev)

        ctx = ctx.to(dev)
        means = []
        for i in range(0, ctx.shape[0], self.series_chunk):
            _q, mean = self._pipe.predict_quantiles(
                ctx[i:i + self.series_chunk], prediction_length=self.pred_length
            )
            means.append(mean[:, 0])  # first forecast step
        # `predict_quantiles` returns CPU tensors regardless of the input device.
        return torch.cat(means, dim=0).reshape(b, d).to(device=dev, dtype=self._dev_anchor.dtype)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_extra = self._extra_input_slice(x)           # [B, L, extra_input_dim]
        feats = self._channel_forecasts(x_extra)        # [B, extra_input_dim], detached
        return feats.mean(dim=1, keepdim=True)           # [B, 1]  ("mean"; no-op when D==1)

    # -- task hooks --------------------------------------------------------------
    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        # point forecast of the next-step log-return; higher == more bullish
        return self._sign * raw.reshape(raw.shape[0])

    def compute_loss(self, raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        # Never backpropped — the Trainer runs this model eval-only (zero_shot=True, no
        # trainable params). Which batch field to compare against is a config choice
        # (`loss_target`), not hardcoded: default "fwd_logret" is correct BY
        # CONSTRUCTION because `forward` is engineered to output a next-step log-return
        # in that exact scale/horizon (see module docstring) — "y" is only meaningful if
        # the fragment also repoints `data.target_cols` at something in the same scale.
        key = "y" if self._loss_target == "y" else "fwd_logret"
        target = batch[key].to(raw.device).reshape(raw.shape)
        if self._loss_name == "mse":
            return F.mse_loss(raw, target)
        return F.huber_loss(raw, target, delta=self._huber_delta)
