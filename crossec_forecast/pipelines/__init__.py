"""
High-level experiment pipelines.

These wrap the library primitives (data / models / engine / eval) behind a single
`ExperimentConfig` so that CLI scripts and cluster schedulers stay trivial.

    from crossec_forecast.pipelines import load_experiment, run_train
    cfg, run_dir, logger = load_experiment("experiments/experiment.yaml", ["model.name=lstm"])
    run_train(cfg, run_dir, logger)
"""

from .context import (
    config_hash,
    load_config,
    load_experiment,
    model_build_config,
    resolve_run_dir,
    select_device,
    to_data_config,
    to_train_config,
)
from .tracking import WandbTracker
from .train_pipeline import run_train
from .infer_pipeline import run_infer
from .benchmark_pipeline import run_benchmark

__all__ = [
    "load_config",
    "load_experiment",
    "resolve_run_dir",
    "config_hash",
    "to_data_config",
    "to_train_config",
    "model_build_config",
    "select_device",
    "WandbTracker",
    "run_train",
    "run_infer",
    "run_benchmark",
]
