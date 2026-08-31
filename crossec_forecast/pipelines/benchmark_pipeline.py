"""Run the multi-model BenchmarkEngine from an ExperimentConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..configs.default_config import BenchmarkConfig
from ..data import build_dataloaders
from ..eval import BenchmarkEngine
from ..models import build_model  # noqa: F401  (also populates the registry)
from ..utils.logger import setup_logger
from .context import to_data_config, to_train_config
from .tracking import WandbTracker


def run_benchmark(
    cfg: DictConfig,
    run_dir: Path,
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    """Train + test every model in ``cfg.benchmark.models`` under one data split."""
    logger = logger or setup_logger("pipeline.benchmark")
    run_dir = Path(run_dir)

    data_config = to_data_config(cfg)
    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg.data.path, config=data_config)

    models_config = OmegaConf.to_container(cfg.benchmark.models, resolve=True)
    bench_config = BenchmarkConfig(
        models=models_config,
        top_quantile=float(cfg.benchmark.top_quantile),
        export_dir=str(run_dir / "reports"),
    )

    engine = BenchmarkEngine(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        meta_info=meta,
        train_config=to_train_config(cfg, run_dir / "checkpoints"),
        benchmark_config=bench_config,
        seed=int(cfg.experiment.seed),
        logger=logger,
    )
    summary_df = engine.run()

    tracker = WandbTracker(cfg, run_dir, job_type="benchmark")
    if tracker.enabled:
        try:
            import wandb

            tracker.log({"benchmark/summary": wandb.Table(dataframe=summary_df)})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"wandb table logging failed: {exc}")
        if not summary_df.empty:
            best = summary_df.iloc[0].to_dict()
            tracker.summary({f"best/{k}": v for k, v in best.items() if k != "model"})
            tracker.summary({"best/model": best.get("model")})
    tracker.finish()

    return summary_df
