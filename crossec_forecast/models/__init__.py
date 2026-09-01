import importlib
import logging

from .base import BaseClassifierModel
from .registry import (
    register_model,
    build_model,
    list_registered_models,
    is_model_registered,
    is_model_available,
    list_available_models,
)
from ._optional import ModelDependencyError, module_available, require_modules

# --- Pure-python models --------------------------------------------------------------
# No optional dependencies. A failure here is a real bug, so let it raise.
from .mlp import MLPClassifier
from .lstm import LSTMClassifier
from .dlinear import DLinearClassifier
from .pretrained import PretrainedBackboneModel

_log = logging.getLogger(__name__)

# --- Optional-backend wrappers -----------------------------------------------------
# Each of these SHOULD keep its heavy `import` inside `__init__` (via
# `_optional.require_modules`) so the module imports — and the model registers — even
# when its backend library is absent from this interpreter. `build_model` then raises
# a clean ``ModelDependencyError`` only when that specific model is actually requested.
#
# This loop is a safety net: if a wrapper module fails to import for ANY reason (a
# stray top-level heavy import, a version-incompat error at import, ...), it degrades
# to "that one model is unavailable" + a warning, instead of breaking
# ``import crossec_forecast`` for every model — including the ones this interpreter
# could run. Add new TSFM wrappers here by module name.
_OPTIONAL_WRAPPER_MODULES: tuple = (
    # "chronos_bolt",
    # "moment",
)

for _modname in _OPTIONAL_WRAPPER_MODULES:
    try:
        importlib.import_module(f".{_modname}", __name__)
    except Exception as exc:  # noqa: BLE001 - defensive: never let one wrapper break import
        _log.warning(
            "model wrapper '%s' failed to import (%s: %s) — that model is unavailable "
            "in this interpreter; all other models are unaffected.",
            _modname, type(exc).__name__, exc,
        )

__all__ = [
    "BaseClassifierModel",
    "register_model",
    "build_model",
    "list_registered_models",
    "is_model_registered",
    "is_model_available",
    "list_available_models",
    "ModelDependencyError",
    "module_available",
    "require_modules",
    "MLPClassifier",
    "LSTMClassifier",
    "DLinearClassifier",
    "PretrainedBackboneModel",
]
