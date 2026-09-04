"""
Chronos-Bolt, zero-shot — Pattern B (native forecast head) x NO training: the frozen
backbone's own next-step forecast, read directly as the cross-sectional score.

Registered as ``chronos_bolt_zeroshot``. Unlike ``chronos_bolt_head_only`` there is no
trainable head and no per-channel-then-linear-probe compromise — this is Chronos-Bolt run
exactly the way it is meant to be used: feed it a univariate context window, read its own
one-step forecast, done.

Recommended usage — point the input at an actual return series, not the 24-feature panel:

  data:
    feature_cols: [crossec_logret_1_mad_Zscore]   # <- a single return-like column, D=1

so ``x [B, L, D=1]`` literally *is* the series Chronos expects. See
``experiments/models/chronos_bolt_zeroshot.yaml``. The wrapper does not hardcode D==1 —
it forecasts each of the D channels independently (frozen, no grad) and reduces them with
a fixed, non-trainable rule (mean), so a D>1 config still runs (degrading gracefully to
the "route B" cross-channel probe) — but the honest, native-usage zero-shot experiment is
the D==1 case, where the reduction is a no-op.

  x [B, L, D] --(per-channel, frozen Chronos-Bolt)--> 1-step forecast [B, D]
             --(mean across D)--> raw [B, 1]
             --(to_score: * sign)--> score [B]      (higher == more bullish)

  output_kind = "point_forecast"
  zero_shot   = True   -> Trainer runs this eval-only (no optimizer, no training loop)
  compute_loss -> Huber/MSE of raw vs the realized `fwd_logret_1` (batch["fwd_logret"],
                  NOT batch["y"]/target_col — so no `data.target_col` override is needed
                  in the config fragment). Never backpropped; reported as val_loss only.

Backend: `chronos-forecasting` (import name `chronos`). Recent Python only
(`pip install -e ".[chronos]"`).

config keys (model.config):
  model_id            "amazon/chronos-bolt-{tiny,mini,small,base}"  (default: small)
  context_length      int | null   null = use the full data.seq_len window
  prediction_length   int          default 1 (next bar)
  reduction           "mean"       how the D per-channel forecasts collapse to one raw
                       scalar (only "mean" is implemented; a no-op when D==1)
  sign                +1.0 | -1.0  flip the factor if val Rank IC comes out negative
  loss_type           "huber" | "mse"   (default huber; diagnostic val_loss only)
  huber_delta         float        default 1.0
  series_chunk        int          max series per Chronos call (default 4096)
  torch_dtype         str | null   e.g. "bfloat16"; null = fp32
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

        self.model_id = str(config.get("model_id") or self.pretrained_path or "amazon/chronos-bolt-small")
        self.context_length = config.get("context_length")
        self.pred_length = int(config.get("prediction_length", 1))
        self.series_chunk = int(config.get("series_chunk", 4096))
        self._reduction = str(config.get("reduction", "mean")).lower()
        if self._reduction not in ("mean",):
            raise ValueError("chronos_bolt_zeroshot: reduction must be 'mean'")
        self._sign = float(config.get("sign", 1.0))
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

    # -- frozen Chronos forecast per input channel -----------------------------------
    @torch.no_grad()
    def _channel_forecasts(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, D] -> [B, D] one-step-ahead forecast per channel (detached)."""
        b, L, d = x.shape
        ctx = x.permute(0, 2, 1).reshape(b * d, L)
        if self.context_length:
            ctx = ctx[:, -int(self.context_length):]

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
        feats = self._channel_forecasts(x)            # [B, D], detached (frozen backbone)
        return feats.mean(dim=1, keepdim=True)         # [B, 1]  ("mean"; no-op when D==1)

    # -- task hooks --------------------------------------------------------------
    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        # point forecast of the return series; higher == more bullish
        return self._sign * raw.reshape(raw.shape[0])

    def compute_loss(self, raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        # Never backpropped — the Trainer runs this model eval-only (zero_shot=True, no
        # trainable params). Compared against the realized forward return directly
        # (batch["fwd_logret"], not batch["y"]/target_col) so this wrapper needs no
        # `data.target_col` override — only `data.feature_cols` changes vs the base config.
        target = batch["fwd_logret"].to(raw.device).reshape(raw.shape)
        if self._loss_name == "mse":
            return F.mse_loss(raw, target)
        return F.huber_loss(raw, target, delta=self._huber_delta)
