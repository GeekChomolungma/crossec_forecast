"""
Extension point for wrapping an externally *pretrained* time-series backbone
(e.g. Chronos2, TimesFM, Moirai) into the same plug-and-play model contract used
by the from-scratch models (mlp / lstm / dlinear).

v1 of this project only trains from-scratch models on our own panel data — nothing
here loads real weights yet, and nothing here is registered, so it does not show up
in `list_registered_models()` and cannot be selected via `model.name` in a config.
It exists purely so a future pretrained backbone has a defined slot to plug into
instead of bolting one on ad hoc (the mistake `tsfm_wrapper.py` made: it *claimed*
to be a foundation-model adapter but was actually just a from-scratch nn.Transformer
with no pretrained weights anywhere — misleading, so it was removed).

To wire in a real pretrained backbone later:

    @register_model("chronos2")
    class Chronos2Wrapper(PretrainedBackboneModel):
        def __init__(self, config):
            super().__init__(config)
            self.backbone = ...  # actually load a checkpoint/hub id here, e.g.
                                  # from self.pretrained_path, and honor self.freeze_backbone
            self.head = ...      # small trainable head on top, [*, ...] -> [B, 1]

        def forward(self, x, **kwargs):
            ...

That's it — `build_model()` / the registry / sweep configs all dispatch on the
registered name generically, so `model.name=chronos2` then works exactly like
`model.name=lstm` does today, no other wiring required.
"""
from typing import Dict, Any
import torch
from .base import BaseClassifierModel


class PretrainedBackboneModel(BaseClassifierModel):
    """
    Base class for models that wrap an externally pretrained time-series backbone.

    Subclasses are expected to:
      * load real pretrained weights in `__init__` (checkpoint path / hub id read
        from `config`, e.g. `self.pretrained_path`),
      * implement `forward` against the shared [B, seq_len, feature_dim] -> [B, 1]
        contract (project/pool the backbone's output through a small trainable head),
      * decide their own fine-tuning strategy via `self.freeze_backbone`.

    Deliberately NOT decorated with @register_model: instantiating this base class
    directly is a programming error, not a usable model — it stays out of the
    registry until a real backbone is wired into a subclass.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pretrained_path = config.get("pretrained_path")
        self.freeze_backbone = bool(config.get("freeze_backbone", True))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "PretrainedBackboneModel is an interface, not a runnable model. Subclass "
            "it, load real pretrained weights in __init__, implement forward(), and "
            "register the subclass with @register_model(...) before selecting it via "
            "model.name in a config."
        )
