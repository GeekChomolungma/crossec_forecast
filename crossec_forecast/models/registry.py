import sys
from typing import Dict, Type, Optional, List, Callable, Any

from .base import BaseClassifierModel
from ._optional import ModelDependencyError, module_available

_MODEL_REGISTRY: Dict[str, Type[BaseClassifierModel]] = {}


def register_model(name: Optional[str] = None) -> Callable[[Type[BaseClassifierModel]], Type[BaseClassifierModel]]:
    """
    Decorator to register a model class into the global Model Registry.
    
    Usage:
        @register_model("my_model")
        class MyModel(BaseClassifierModel):
            ...
    """
    def decorator(cls: Type[BaseClassifierModel]) -> Type[BaseClassifierModel]:
        model_name = name or cls.__name__.lower()
        if model_name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' is already registered to {_MODEL_REGISTRY[model_name]}.")
        _MODEL_REGISTRY[model_name] = cls
        return cls

    return decorator


def _missing_backends(model_cls: Type[BaseClassifierModel]) -> List[str]:
    return [m for m in getattr(model_cls, "REQUIRED_MODULES", ()) if not module_available(m)]


def build_model(model_name: str, config: Dict[str, Any]) -> BaseClassifierModel:
    """
    Factory function to instantiate a model by its registered name and config dictionary.

    Raises ``ModelDependencyError`` (a subclass of ``ImportError``) if the model is
    registered but its backend library is not importable in this interpreter — the
    common case when TSFM libs are split across venvs by Python-version conflicts.
    """
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(list_registered_models())
        raise KeyError(f"Model '{model_name}' not found in registry. Available models: [{available}]")

    model_cls = _MODEL_REGISTRY[key]
    missing = _missing_backends(model_cls)
    if missing:
        hint = getattr(model_cls, "PYTHON_HINT", "") or ""
        hint_line = f" Backend needs: {hint}." if hint else ""
        raise ModelDependencyError(
            f"Model '{model_name}' is registered but its backend package(s) {missing} are "
            f"not importable in this interpreter (Python {sys.version_info.major}."
            f"{sys.version_info.minor}). Install the matching extra from pyproject "
            f"optional-dependencies and run the model from a compatible interpreter.{hint_line}"
        )
    return model_cls(config)


def list_registered_models() -> List[str]:
    """List all currently registered model names (regardless of backend availability)."""
    return sorted(list(_MODEL_REGISTRY.keys()))


def is_model_registered(model_name: str) -> bool:
    """Check if a model name is registered."""
    return model_name.lower() in _MODEL_REGISTRY


def is_model_available(model_name: str) -> bool:
    """
    True if the model is registered AND every entry in its ``REQUIRED_MODULES`` imports
    in this interpreter. Pure-python models (empty ``REQUIRED_MODULES``) are always
    available; a wrapper whose TSFM library lives in another venv is not.
    """
    model_cls = _MODEL_REGISTRY.get(model_name.lower())
    if model_cls is None:
        return False
    return not _missing_backends(model_cls)


def list_available_models() -> List[str]:
    """Registered models whose backend libraries are importable in this interpreter."""
    return sorted(n for n in _MODEL_REGISTRY if is_model_available(n))

