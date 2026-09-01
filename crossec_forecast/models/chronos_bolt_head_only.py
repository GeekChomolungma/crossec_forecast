"""
Chronos-Bolt, head-only — Pattern B (native forecast head) × frozen backbone + linear head.

Registered as ``chronos_bolt_head_only`` (the name spells out the paradigm: this is NOT a
full Chronos-Bolt model, it is a linear probe on its zero-shot forecasts).

Chronos-Bolt is a univariate zero-shot forecaster. We run it **frozen** on each of the D
z-scored feature channels of the lookback window to get D one-step-ahead forecasts, then
a small trainable head maps those D forecasts to a predicted next-bar return
(`fwd_logret_1`).

  output_kind = "point_forecast"
  to_score(raw)      -> the predicted return (higher == more bullish); drives Rank IC
  compute_loss       -> Huber / MSE vs the configured target column (batch["y"])
  model selection    -> unchanged: val mean Rank IC

The frozen Chronos pipeline is held as a plain attribute (not an nn submodule), so its
weights never enter the optimizer / grad-clip and checkpoints stay head-only.

Backend: `chronos-forecasting` (import name `chronos`). Recent Python only
(`pip install -e ".[chronos]"`). Registers even when the backend is absent; `build_model`
then raises ModelDependencyError.

config keys (model.config):
  model_id            "amazon/chronos-bolt-{tiny,mini,small,base}"  (default: small)
  context_length      int | null   null = use the full data.seq_len window
  prediction_length   int          default 1 (next bar)
  head_hidden         int | null   null = LayerNorm+Linear(D,1); int = + a hidden layer
  loss_type           "huber" | "mse"   (default huber)
  huber_delta         float        default 1.0
  series_chunk        int          max series per Chronos call (default 4096)
  torch_dtype         str | null   e.g. "bfloat16"; null = fp32
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained import PretrainedBackboneModel
from .registry import register_model
from ._optional import require_modules


@register_model("chronos_bolt_head_only")
class ChronosBoltHeadOnly(PretrainedBackboneModel):
    REQUIRED_MODULES = ("chronos",)
    PYTHON_HINT = 'recent Python; pip install -e ".[chronos]"'
    output_kind = "point_forecast"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # sets self.pretrained_path, self.freeze_backbone
        chronos = require_modules(
            "chronos_bolt_head_only", self.REQUIRED_MODULES,
            extra="chronos", python_hint=self.PYTHON_HINT,
        )["chronos"]

        self.model_id = str(config.get("model_id") or self.pretrained_path or "amazon/chronos-bolt-small")
        self.context_length = config.get("context_length")
        self.pred_length = int(config.get("prediction_length", 1))
        self.series_chunk = int(config.get("series_chunk", 4096))
        self._loss_name = str(config.get("loss_type", "huber")).lower()
        self._huber_delta = float(config.get("huber_delta", 1.0))

        if not self.freeze_backbone:
            raise NotImplementedError(
                "chronos_bolt_head_only supports freeze_backbone=True only (the Bolt predict "
                "API is inference-only). Fine-tuning the backbone is future work — see TO_IMPROVE.md."
            )

        from_kwargs: Dict[str, Any] = {}
        torch_dtype = config.get("torch_dtype")
        if torch_dtype:
            from_kwargs["torch_dtype"] = getattr(torch, str(torch_dtype))
        # Plain attribute, NOT registered as a submodule (keeps it out of parameters()).
        self._pipe = chronos.BaseChronosPipeline.from_pretrained(self.model_id, **from_kwargs)
        for p in self._pipe.model.parameters():
            p.requires_grad_(False)
        self._pipe.model.eval()

        # Trainable head: D per-channel one-step forecasts -> predicted next-bar return.
        head_hidden = config.get("head_hidden")
        if head_hidden:
            h = int(head_hidden)
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, h), nn.GELU(),
                nn.Linear(h, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 1),
            )

    # -- Chronos forecast per feature channel -----------------------------------------
    @torch.no_grad()
    def _channel_forecasts(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, D] -> [B, D] one-step-ahead forecast per channel (detached)."""
        b, L, d = x.shape
        ctx = x.permute(0, 2, 1).reshape(b * d, L)
        if self.context_length:
            ctx = ctx[:, -int(self.context_length):]

        dev = self.head[-1].weight.device
        dtype = self.head[-1].weight.dtype
        if next(self._pipe.model.parameters()).device != dev:
            self._pipe.model.to(dev)

        ctx = ctx.to(dev)
        means = []
        for i in range(0, ctx.shape[0], self.series_chunk):
            _q, mean = self._pipe.predict_quantiles(
                ctx[i:i + self.series_chunk], prediction_length=self.pred_length
            )
            means.append(mean[:, 0])  # first forecast step
        # `predict_quantiles` returns CPU tensors regardless of the input device, so the
        # concatenated result must be moved back to `dev` explicitly (not just cast dtype)
        # before it reaches `self.head`, which lives on `dev`.
        return torch.cat(means, dim=0).reshape(b, d).to(device=dev, dtype=dtype)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feats = self._channel_forecasts(x)   # [B, D], detached (frozen backbone)
        return self.head(feats)              # [B, 1] -> predicted fwd_logret_1

    # -- task hooks ------------------------------------------------------------------
    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        # point forecast of the forward return; higher == more bullish
        return raw.reshape(raw.shape[0])

    def compute_loss(self, raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        target = batch["y"].to(raw.device).reshape(raw.shape)
        if self._loss_name == "mse":
            return F.mse_loss(raw, target)
        return F.huber_loss(raw, target, delta=self._huber_delta)
