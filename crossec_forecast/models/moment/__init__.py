"""
MOMENT wrapper family. Each submodule is one adaptation paradigm for the same backbone —
see the module docstrings in ``head_only.py`` / ``zeroshot.py`` and
``models/pretrained_research.md`` for the A/B/C pattern taxonomy.

Not imported eagerly here: ``models/__init__.py`` imports each submodule individually
(via ``_OPTIONAL_WRAPPER_MODULES``) so one broken/absent backend never takes the other
down with it.
"""
