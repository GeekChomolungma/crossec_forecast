"""
Optional-backend plumbing for model plugins.

Some wrappers depend on a third-party library (Chronos, MOMENT, TimesFM, ...) whose
dependency pins conflict — e.g. ``chronos-forecasting`` wants a recent Python while
``momentfm``'s older ``transformers`` pin wants Python <=3.11. They cannot coexist in one
interpreter, so their backends get installed into separate environments (named however
the user likes) and models are classified by the Python version they need.

The contract that makes this survivable:

  * ``import crossec_forecast`` must work in **any** environment, missing backends or not.
  * A wrapper whose backend is absent still **registers** (shows in
    ``list_registered_models()``, can be named in a config) but **fails to instantiate**
    with a clear, actionable error instead of an ``ImportError`` at module load.
  * ``list_available_models()`` / ``is_model_available()`` report what is actually runnable
    here, and ``BenchmarkEngine`` skips the rest with a warning.

Wrapper pattern::

    from .base import BaseClassifierModel
    from .registry import register_model
    from ._optional import require_modules

    @register_model("moment")
    class MomentWrapper(BaseClassifierModel):
        REQUIRED_MODULES = ("momentfm",)          # checked by the registry / benchmark
        PYTHON_HINT = "Python 3.9-3.11"           # shown in the error; NOT an env name
        output_kind = "binary_prob"

        def __init__(self, config):
            super().__init__(config)
            mods = require_modules(                # raises ModelDependencyError if absent
                "moment", self.REQUIRED_MODULES, extra="moment",
                python_hint=self.PYTHON_HINT,
            )
            momentfm = mods["momentfm"]
            self.backbone = momentfm.MOMENTPipeline.from_pretrained(...)
            ...

Environment names are user-defined, so nothing here hardcodes one — models are classified
by the Python version their backend needs (`PYTHON_HINT`), and the error tells you to run
from *an* interpreter that satisfies it.
"""
from __future__ import annotations

import importlib
import sys
from typing import Dict, Iterable


class ModelDependencyError(ImportError):
    """A registered model plugin cannot run because a backend package is not importable
    in the current Python environment (usually a deliberate cross-venv split)."""


def _pyver() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def module_available(name: str) -> bool:
    """True if ``import name`` succeeds in this interpreter (cheap, result not cached)."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def require_modules(
    model_name: str,
    modules: Iterable[str],
    *,
    extra: str | None = None,
    python_hint: str | None = None,
) -> Dict[str, object]:
    """
    Import every name in ``modules`` and return ``{name: module}``.

    On any failure raise :class:`ModelDependencyError` with an actionable message
    (which extra to install, which Python version the backend needs). Call this at the
    top of a wrapper's ``__init__`` — it both guards and hands back the module objects.

    ``python_hint`` is a free-text version constraint (e.g. ``"Python 3.9-3.11"``), never
    an environment name — envs are the user's to name.
    """
    imported: Dict[str, object] = {}
    failures = []
    for name in modules:
        try:
            imported[name] = importlib.import_module(name)
        except Exception as exc:  # ImportError, or version-incompat errors on import
            failures.append((name, exc))

    if failures:
        names = ", ".join(repr(n) for n, _ in failures)
        lines = [
            f"Model '{model_name}' needs {names}, which is not importable in this "
            f"interpreter (Python {_pyver()}, {sys.prefix}).",
        ]
        if extra:
            lines.append(f'  install:  pip install -e ".[{extra}]"')
        if python_hint:
            lines.append(f"  run it from an interpreter that satisfies: {python_hint}")
        lines.append(f"  first import error: {failures[0][1]!r}")
        raise ModelDependencyError("\n".join(lines))

    return imported
