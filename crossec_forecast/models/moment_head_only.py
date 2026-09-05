"""
MOMENT, head-only — Pattern A (representation) x frozen encoder + linear probe.

Registered as ``moment_head_only`` (the name spells out the paradigm: NOT a full MOMENT
model, a linear probe on its frozen encoder embeddings).

MOMENT is a T5-encoder masked-reconstruction time-series foundation model. Here its
encoder is frozen and used purely as a feature extractor:

  x [B, L, D]  --(permute)-->  [B, D, seq_len]  (left-padded / truncated to MOMENT's
  fixed 512-step context, with an input_mask)
             --MOMENT.embed(reduction="mean")-->  window embedding [B, d_model]
             --trainable head-->  [B, 1]

  task="classification" (default): output_kind="binary_prob" -> sigmoid, BCE/Focal on
      `logret1_win`; sits next to mlp / lstm / dlinear on the exact same task.
  task="regression": output_kind="point_forecast" -> Huber/MSE vs `fwd_logret_1`
      (then set `data.target_cols: [fwd_logret_1]` in the fragment).

Frozen MOMENT is a plain attribute (not an nn submodule): the optimizer and the
checkpoint stay head-only. LoRA / unfreezing needs Trainer changes — see
pretrained_research.md section 8.

Backend: `momentfm`. Python 3.9-3.11 (older transformers pin); `pip install -e ".[moment]"`.

config keys (model.config):
  model_id     "AutonLab/MOMENT-1-{small,base,large}"   (default: small)
  task         "classification" (default) | "regression"
  head_hidden  int | null   null = LayerNorm + Linear(d_model, 1); int = + a hidden layer
  loss_type    regression only: "huber" | "mse"  (default huber)
  huber_delta  float  default 1.0
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained import PretrainedBackboneModel
from .registry import register_model
from ._optional import require_modules


@register_model("moment_head_only")
class MomentHeadOnly(PretrainedBackboneModel):
    REQUIRED_MODULES = ("momentfm",)
    PYTHON_HINT = 'Python 3.9-3.11; pip install -e ".[moment]"'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        momentfm = require_modules(
            "moment_head_only", self.REQUIRED_MODULES,
            extra="moment", python_hint=self.PYTHON_HINT,
        )["momentfm"]

        self.model_id = str(config.get("model_id") or self.pretrained_path or "AutonLab/MOMENT-1-small")
        task = str(config.get("task", "classification")).lower()
        if task not in ("classification", "regression"):
            raise ValueError("moment_head_only: task must be 'classification' or 'regression'")
        self.task = task
        self.output_kind = "binary_prob" if task == "classification" else "point_forecast"
        self._loss_name = str(config.get("loss_type", "huber")).lower()
        self._huber_delta = float(config.get("huber_delta", 1.0))

        if not self.freeze_backbone:
            raise NotImplementedError(
                "moment_head_only supports freeze_backbone=True only. LoRA / unfreezing "
                "needs the Trainer optimizer-hook work — see pretrained_research.md section 8."
            )

        # NB: no force_download here. A sweep runs N jobs concurrently against one
        # shared HF cache on /projects (NFS); force_download makes every job re-fetch
        # model.safetensors and they clobber each other's blob mid-rename -> truncated
        # file -> SafetensorError "header too small". Pre-warm the cache once on a
        # login node (huggingface-cli download AutonLab/MOMENT-1-{small,base}); jobs
        # then just read it. Set HF_HUB_OFFLINE=1 in the .sbatch to force cache-only.
        _moment = momentfm.MOMENTPipeline.from_pretrained(
            self.model_id, model_kwargs={"task_name": "embedding"},
        )
        _moment.init()
        for p in _moment.parameters():
            p.requires_grad_(False)
        _moment.eval()
        try:  # gradient checkpointing is pointless (and noisy) for a frozen forward
            _moment.encoder.gradient_checkpointing_disable()
        except Exception:
            pass
        # Store WITHOUT registering as a submodule: MOMENTPipeline is an nn.Module, so a
        # plain `self._moment = ...` would auto-register it and pull its ~35M frozen params
        # into parameters() / state_dict() (optimizer bloat, huge checkpoints). object's
        # __setattr__ bypasses nn.Module's. `to()` / `eval()` are handled manually.
        object.__setattr__(self, "_moment", _moment)
        self._moment_dev = next(_moment.parameters()).device
        self.moment_seq_len = int(_moment.config.seq_len)
        d_model = int(_moment.config.d_model)

        head_hidden = config.get("head_hidden")
        if head_hidden:
            h = int(head_hidden)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, h), nn.GELU(),
                nn.Linear(h, 1),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    # -- frozen MOMENT encoder embedding -------------------------------------------
    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x [B, L, D] -> [B, d_model] frozen MOMENT window embedding (detached).

        MOMENT wants a fixed 512-step context. This wrapper targets the L=512 config
        (no padding). Shorter windows are left-padded with zeros and fed with a full
        input_mask — the features are MAD-zscored (~centered), so a zero prefix is a
        benign flat history, and it avoids MOMENT's patch-mask edge cases when the valid
        region is not a whole number of 8-step patches. Longer windows are truncated to
        the most recent 512 steps.
        """
        b, L, _d = x.shape
        S = self.moment_seq_len
        dev = self.head[-1].weight.device
        if self._moment_dev != dev:
            self._moment.to(dev)
            self._moment_dev = dev

        xc = x.permute(0, 2, 1).to(dev)  # [B, D, L]
        if L < S:
            xc = F.pad(xc, (S - L, 0))   # left-pad the older side with zeros
        elif L > S:
            xc = xc[:, :, -S:]
        mask = torch.ones(b, S, device=dev)

        out = self._moment.embed(x_enc=xc, input_mask=mask, reduction="mean")
        return out.embeddings.to(self.head[-1].weight.dtype)  # [B, d_model]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.head(self._embed(x))  # [B, 1]

    # -- task hooks --------------------------------------------------------------
    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        if self.output_kind == "binary_prob":
            return super().to_score(raw)          # sigmoid -> [B]
        return raw.reshape(raw.shape[0])          # point forecast

    def compute_loss(self, raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        if self.output_kind == "binary_prob":
            return super().compute_loss(raw, batch)   # BCE / Focal on batch["y"]
        target = batch["y"].to(raw.device).reshape(raw.shape)
        if self._loss_name == "mse":
            return F.mse_loss(raw, target)
        return F.huber_loss(raw, target, delta=self._huber_delta)
