from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseClassifierModel(nn.Module, ABC):
    """
    Abstract Base Class for every model plugged into the cross-sectional benchmark.

    The forward contract stays ``forward(x [B, L, D], **kwargs) -> raw``. What "raw"
    means is model-specific and declared via ``output_kind``; two hooks turn that raw
    output into the two things the engine actually needs:

      * ``to_score(raw)``    -> a 1-D cross-sectional ranking signal ``[B]`` (higher ==
                                more bullish). This is the ONLY thing Rank IC / ICIR /
                                long-short consume, so it is the single quantity that
                                makes heterogeneous models comparable.
      * ``compute_loss(raw, batch)`` -> the scalar training loss for this model. The
                                model sees the whole batch and picks whichever target
                                column it needs, so the Trainer is loss-agnostic.

    The three from-scratch models (mlp / lstm / dlinear) are ``output_kind ==
    "binary_prob"`` and inherit the defaults below (sigmoid + BCE/Focal on ``batch["y"]``).
    A future pretrained / forecasting backbone overrides ``output_kind`` and both hooks.
    """

    # "binary_prob" | "point_forecast" | "quantile" | "embedding" | ...
    # Downstream (metrics gating, artifact schema) branches on this.
    output_kind: str = "binary_prob"

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.seq_len = int(config.get("seq_len", 6))
        self.feature_dim = int(config.get("feature_dim", 24))
        self.num_classes = int(config.get("num_classes", 1))
        self._loss_fn = None  # lazily built by the default compute_loss

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, seq_len, feature_dim]

        Returns:
            Raw output. For ``output_kind == "binary_prob"`` this is logits [B, 1].
        """
        pass

    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Map raw ``forward`` output to a 1-D cross-sectional ranking score ``[B]``.

        Contract: **higher == more bullish** (expected to rank higher in forward
        return), so cross-sectional Rank IC comes out with a consistent sign.

        Default (``output_kind == "binary_prob"``): raw is logits [B, 1] ->
        ``sigmoid`` -> P(beat cross-sectional median), flattened to [B].
        """
        return torch.sigmoid(raw).reshape(raw.shape[0])

    def compute_loss(self, raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Scalar training loss for this model.

        Given the raw ``forward`` output and the full collated ``batch`` (so a subclass
        can pick whichever target column it needs). Default = binary cross-entropy /
        focal on ``batch["y"]``, selected via model-config keys ``loss_type``
        ("bce" | "focal"), ``focal_gamma``, ``focal_alpha``.
        """
        if self._loss_fn is None:
            from ..engine.losses import get_loss_fn

            self._loss_fn = get_loss_fn(
                loss_type=str(self.config.get("loss_type", "bce")),
                gamma=float(self.config.get("focal_gamma", 2.0)),
                alpha=self.config.get("focal_alpha"),
            )
        target = batch["y"].to(raw.device)
        return self._loss_fn(raw, target)

    def predict_proba(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Deprecated compatibility shim: a [B, 1] probability-shaped tensor.

        New engine code paths call ``to_score`` directly. Kept so external callers and
        tests that expect ``predict_proba`` keep working; only meaningful for
        ``output_kind == "binary_prob"``.
        """
        return self.to_score(self.forward(x, **kwargs)).reshape(-1, 1)
