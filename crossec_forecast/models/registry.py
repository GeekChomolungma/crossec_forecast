from typing import Dict, Type, Optional, List, Callable, Any
from .base import BaseClassifierModel

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


def build_model(model_name: str, config: Dict[str, Any]) -> BaseClassifierModel:
    """
    Factory function to instantiate a model by its registered name and config dictionary.
    """
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(list_registered_models())
        raise KeyError(f"Model '{model_name}' not found in registry. Available models: [{available}]")
    
    model_cls = _MODEL_REGISTRY[key]
    return model_cls(config)


def list_registered_models() -> List[str]:
    """List all currently registered model names."""
    return sorted(list(_MODEL_REGISTRY.keys()))


def is_model_registered(model_name: str) -> bool:
    """Check if a model name is registered."""
    return model_name.lower() in _MODEL_REGISTRY

