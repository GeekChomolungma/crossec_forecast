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

    # Backend packages this plugin needs importable to run (empty == pure-python,
    # always available). The registry / BenchmarkEngine use this to skip a plugin
    # whose library is not in the current interpreter (TSFM libs get split across
    # interpreters by Python-version conflicts); the wrapper's __init__ should still
    # call models._optional.require_modules(...) for the actionable error + handles.
    REQUIRED_MODULES: tuple = ()
    # Free-text Python-version constraint of the backend, surfaced in the "can't build
    # here" error (e.g. "Python 3.9-3.11"). Never an environment name — envs are the
    # user's to name; models are classified by the interpreter version they need.
    PYTHON_HINT: str = ""

    # Set True by a wrapper with NO trainable parameters — a pure zero-shot baseline:
    # run the pretrained backbone as-is, map its output to a score, and compare it on
    # val / test Rank IC next to the trained models. The Trainer then skips the
    # optimizer, scheduler and training loop and scores the model once (see
    # ``Trainer._fit_eval_only``). A model that exposes no ``requires_grad`` parameters
    # is treated the same way even without setting this flag; the flag also lets a
    # wrapper that *does* carry a (frozen) head opt into the same eval-only path.
    zero_shot: bool = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.seq_len = int(config.get("seq_len", 6))
        self.feature_dim = int(config.get("feature_dim", 24))
        self.num_classes = int(config.get("num_classes", 1))
        self._loss_fn = None  # lazily built by the default compute_loss

    def trainable_parameters(self):
        """
        Parameters the Trainer should hand to the optimizer. Default: every parameter
        with ``requires_grad=True``.

        Frozen-backbone wrappers already keep the backbone out of ``parameters()`` (a
        non-submodule attribute), so this returns just the head. A pure zero-shot model
        returns ``[]`` and the Trainer runs it eval-only. A future LoRA wrapper can
        override this to return only its adapter + head tensors.
        """
        return [p for p in self.parameters() if p.requires_grad]

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
