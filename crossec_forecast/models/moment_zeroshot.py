"""
MOMENT, zero-shot — Pattern A (representation) x NO training: a reconstruction-error
factor read straight off the frozen, pretrained model.

Registered as ``moment_zeroshot``. There is no trainable head and no fine-tuning. The
only genuinely *pretrained* capability MOMENT exposes is masked reconstruction
("classification and forecasting heads must be fine-tuned"), so this wrapper scores each
lookback window by how badly the frozen model reconstructs it and uses that scalar as a
cross-sectional signal — the cheap "is there any signal here at all?" probe before
committing to head-tuning / LoRA.

  x [B, L, D] --(permute / pad-or-truncate to MOMENT's 512-step context)--> [B, D, 512]
             --MOMENT.detect_anomalies (full mask, criterion=mse)--> per-(channel,step)
               squared reconstruction error [B, D, 512]
             --masked mean over valid steps & channels--> reconstruction error [B, 1]

  output_kind = "anomaly_score"
  to_score(raw) -> sign * error   (contract: higher == more bullish. Which *sign* of
                   reconstruction error predicts the forward return is exactly what this
                   zero-shot run measures — flip ``sign`` if val Rank IC comes out < 0.)
  compute_loss  -> mean batch reconstruction error, reported as ``val_loss`` only. It is
                   never backpropped: with no trainable params + ``zero_shot = True`` the
                   Trainer runs this model eval-only (see Trainer._fit_eval_only).

Frozen MOMENT is held as a NON-submodule attribute (like ``moment_head_only``) so its
~35M frozen params stay out of ``parameters()`` / ``state_dict()``; ``to()`` is handled
manually via a tiny device-anchor buffer.

Backend: `momentfm`. Python 3.9-3.11 (older transformers pin); `pip install -e ".[moment]"`.

config keys (model.config):
  model_id           "AutonLab/MOMENT-1-{small,base,large}"   (default: small)
  anomaly_criterion  "mse" | "mae"    (default: mse)
  sign               +1.0 | -1.0      (default: +1.0) — factor direction
"""
from typing import Dict, Any

import torch
import torch.nn.functional as F

from .pretrained import PretrainedBackboneModel
from .registry import register_model
from ._optional import require_modules


@register_model("moment_zeroshot")
class MomentZeroShot(PretrainedBackboneModel):
    REQUIRED_MODULES = ("momentfm",)
    PYTHON_HINT = 'Python 3.9-3.11; pip install -e ".[moment]"'
    output_kind = "anomaly_score"
    zero_shot = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # sets self.pretrained_path, self.freeze_backbone
        momentfm = require_modules(
            "moment_zeroshot", self.REQUIRED_MODULES,
            extra="moment", python_hint=self.PYTHON_HINT,
        )["momentfm"]

        self.model_id = str(config.get("model_id") or self.pretrained_path or "AutonLab/MOMENT-1-small")
        self._crit = str(config.get("anomaly_criterion", "mse")).lower()
        if self._crit not in ("mse", "mae"):
            raise ValueError("moment_zeroshot: anomaly_criterion must be 'mse' or 'mae'")
        self._sign = float(config.get("sign", 1.0))

        if not self.freeze_backbone:
            raise NotImplementedError(
                "moment_zeroshot is a frozen zero-shot baseline; set freeze_backbone=True. "
                "To actually train MOMENT use moment_head_only (linear probe) or a future "
                "LoRA wrapper — see pretrained_research.md section 8."
            )

        # Default task is reconstruction (the one pretrained head); init() is then a no-op.
        _moment = momentfm.MOMENTPipeline.from_pretrained(
            self.model_id, model_kwargs={"task_name": "reconstruction"},
        )
        _moment.init()
        for p in _moment.parameters():
            p.requires_grad_(False)
        _moment.eval()
        try:  # gradient checkpointing is pointless (and noisy) for a frozen forward
            _moment.encoder.gradient_checkpointing_disable()
        except Exception:
            pass
        # Store WITHOUT registering as a submodule (MOMENTPipeline is an nn.Module, so a
        # plain assignment would auto-register it and pull its frozen params into
        # parameters() / state_dict()). object's __setattr__ bypasses nn.Module's.
        object.__setattr__(self, "_moment", _moment)
        self._moment_dev = next(_moment.parameters()).device
        self.moment_seq_len = int(_moment.config.seq_len)

        # The wrapper carries no parameters; this 1-element buffer gives ``model.to(dev)``
        # something to move and lets _recon_error read the Trainer's target device back.
        self.register_buffer("_dev_anchor", torch.zeros(1), persistent=True)

    # -- frozen MOMENT reconstruction-error factor ---------------------------------
    @torch.no_grad()
    def _recon_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        x [B, L, D] -> [B] masked-mean reconstruction error (detached).

        MOMENT wants a fixed 512-step context. L == 512 is the target config (no pad).
        Shorter windows are left-padded with zeros and the pad is marked invalid in
        ``input_mask`` (so the normalizer / attention ignore it) and excluded from the
        error mean. Longer windows are truncated to the most recent 512 steps.
        """
        b, L, _d = x.shape
        S = self.moment_seq_len
        dev = self._dev_anchor.device
        if self._moment_dev != dev:
            self._moment.to(dev)
            self._moment_dev = dev

        xc = x.permute(0, 2, 1).to(dev)          # [B, D, L]
        valid = torch.ones(b, S, device=dev)     # input_mask: 1 = real step
        if L < S:
            xc = F.pad(xc, (S - L, 0))           # left-pad the older side with zeros
            valid[:, : S - L] = 0.0
        elif L > S:
            xc = xc[:, :, -S:]

        out = self._moment.detect_anomalies(
            x_enc=xc, input_mask=valid, anomaly_criterion=self._crit,
        )
        err = out.anomaly_scores                 # [B, D, S] per-(channel, step) error
        v = valid.unsqueeze(1)                   # [B, 1, S]
        denom = v.sum(dim=(1, 2)).clamp_min(1.0) * err.shape[1]
        return (err * v).sum(dim=(1, 2)) / denom  # [B]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._recon_error(x).reshape(-1, 1)  # [B, 1]

    # -- task hooks --------------------------------------------------------------
    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        # higher == more bullish; `sign` flips the factor when val Rank IC is negative
        return self._sign * raw.reshape(raw.shape[0])

    def compute_loss(self, raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        # Never backpropped — the Trainer runs this model eval-only. Surfaced as
        # ``val_loss``: the mean reconstruction error is a useful "how MOMENT-typical is
        # this slice of the panel" diagnostic across the val / test windows.
        return raw.mean() # reduction in all dimensions, [B, 1] -> scalar
