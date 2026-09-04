"""
Experiment context: load YAML -> validated config -> run directory -> seeded state.

Precedence when merging:  dataclass defaults  <  experiment.yaml  <  CLI dot-list overrides
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from ..configs.default_config import DataConfig, TrainConfig
from ..configs.experiment_schema import ExperimentConfig
from ..utils.logger import setup_logger
from ..utils.seed import seed_everything


# --------------------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------------------
def load_config(
    config_path: str | Path | Sequence[str | Path],
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """
    Merge, the right is the winner, from left to right:  structured schema  <-  one or more YAML files  <-  CLI overrides.

    ``config_path`` may be a single path or a list of fragment paths composed in order
    (e.g. a base config plus a per-model fragment).
    """
    paths = [config_path] if isinstance(config_path, (str, Path)) else list(config_path)
    layers = [OmegaConf.structured(ExperimentConfig)]
    for p in paths:
        layers.append(OmegaConf.load(str(p)))
    layers.append(OmegaConf.from_dotlist(list(overrides or [])))
    cfg = OmegaConf.merge(*layers)
    return cfg  # type: ignore[return-value]


def config_hash(cfg: DictConfig, length: int = 8) -> str:
    """Stable short hash of the fully-resolved config (for run-folder names)."""
    payload = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def resolve_run_dir(cfg: DictConfig, *, create: bool = True) -> Path:
    """<output_root>/<experiment.name>/<run.name>/  — auto-naming the run when unset."""
    run_name = cfg.run.name
    if not run_name:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{ts}_{cfg.model.name}_{config_hash(cfg)}"
        cfg.run.name = run_name
    run_dir = Path(cfg.run.output_root) / cfg.experiment.name / run_name
    if create:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    return run_dir


def load_experiment(
    config_path: str | Path | Sequence[str | Path],
    overrides: Optional[Sequence[str]] = None,
    *,
    job_type: str = "train",
    create_run: bool = True,
) -> Tuple[DictConfig, Path, Any]:
    """One-call bootstrap: config + seeded RNG + run dir + logger."""
    cfg = load_config(config_path, overrides)
    seed_everything(int(cfg.experiment.seed))
    run_dir = resolve_run_dir(cfg, create=create_run)
    logger = setup_logger(f"pipeline.{job_type}")
    logger.info(f"Experiment '{cfg.experiment.name}' | job='{job_type}' | run_dir={run_dir}")
    return cfg, run_dir, logger


# --------------------------------------------------------------------------------------
# Adapters: schema node -> existing library dataclasses (kept field-for-field)
# --------------------------------------------------------------------------------------
def _plain(node: Any) -> Any:
    if isinstance(node, (DictConfig,)) or OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def to_data_config(cfg: DictConfig) -> DataConfig:
    d = cfg.data
    feature_cols = _plain(d.feature_cols) if d.feature_cols is not None else None
    cov_cols = _plain(d.cov_cols) if d.cov_cols is not None else None
    return DataConfig(
        target_col=str(d.target_col),
        fwd_ret_col=str(d.fwd_ret_col),
        timestamp_col=str(d.timestamp_col),
        symbol_col=str(d.symbol_col),
        feature_pattern=str(d.feature_pattern),
        feature_cols=feature_cols,
        cov_pattern=(str(d.cov_pattern) if d.cov_pattern is not None else None),
        cov_cols=cov_cols,
        seq_len=int(d.seq_len),
        train_ratio=float(d.split.train_ratio),
        val_ratio=float(d.split.val_ratio),
        test_ratio=float(d.split.test_ratio),
        embargo_steps=int(d.split.embargo_steps),
        batch_size=int(d.batch_size),
        shuffle_train=bool(d.shuffle_train),
        num_workers=int(d.num_workers),
        drop_last=bool(d.drop_last),
    )


def to_train_config(cfg: DictConfig, checkpoint_dir: str | Path) -> TrainConfig:
    t = cfg.train
    return TrainConfig(
        epochs=int(t.epochs),
        lr=float(t.lr),
        weight_decay=float(t.weight_decay),
        grad_clip_norm=float(t.grad_clip_norm),
        early_stopping_patience=int(t.early_stopping_patience),
        min_delta=float(t.min_delta),
        device=str(t.device),
        checkpoint_dir=str(checkpoint_dir),
        scheduler_type=str(t.scheduler_type),
        scheduler_patience=int(t.scheduler_patience),
        scheduler_factor=float(t.scheduler_factor),
    )


def model_build_config(
    cfg_model: DictConfig, *, seq_len: int, feature_dim: int, cov_dim: int = 0
) -> Dict[str, Any]:
    """User model config + auto-injected shape keys, ready for ``build_model``.

    ``cov_dim`` defaults to 0 (no covariates) so callers that don't pass it keep the
    exact same behavior as before covariates existed. ``feature_dim`` / ``cov_dim``
    together describe how a model should slice the packed ``x`` it receives — see
    ``PanelTimeSeriesDataset``'s docstring for the packing convention.
    """
    user_cfg = _plain(cfg_model.config) or {}
    return {
        "seq_len": int(seq_len),
        "feature_dim": int(feature_dim),
        "cov_dim": int(cov_dim),
        "num_classes": 1,
        **user_cfg,
    }


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
