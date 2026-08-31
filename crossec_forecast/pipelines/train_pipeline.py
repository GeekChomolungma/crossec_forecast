"""Train + validate + OOS-test a single model, driven entirely by an ExperimentConfig."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from ..data import build_dataloaders
from ..engine import Trainer
from ..eval import SimpleLongShortBacktester
from ..models import build_model  # noqa: F401  (import also populates the model registry)
from ..utils.logger import setup_logger
from .context import model_build_config, to_data_config, to_train_config
from .tracking import WandbTracker


def run_train(
    cfg: DictConfig,
    run_dir: Path,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the full single-model lifecycle and persist artifacts under ``run_dir``."""
    logger = logger or setup_logger("pipeline.train")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"

    # 1. Data ----------------------------------------------------------------------------
    data_config = to_data_config(cfg)
    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg.data.path, config=data_config)
    logger.info(
        f"Data ready | features={meta['num_features']} seq_len={meta['seq_len']} "
        f"train={meta['n_train_samples']} val={meta['n_val_samples']} test={meta['n_test_samples']}"
    )

    # 2. Model ---------------------------------------------------------------------------
    model_cfg = model_build_config(cfg.model, seq_len=meta["seq_len"], feature_dim=meta["num_features"])
    model = build_model(str(cfg.model.name), model_cfg)

    # 3. Train + validate (wandb logs per-epoch via callback) ---------------------------
    tracker = WandbTracker(cfg, run_dir, job_type="train")
    trainer = Trainer(
        model=model,
        config=to_train_config(cfg, ckpt_dir),
        logger=logger,
        callbacks=[tracker.epoch_callback()],
    )
    fit_res = trainer.fit(train_loader, val_loader)

    # 4. OOS test + backtest -----------------------------------------------------------
    test_metrics = trainer.evaluate(test_loader, top_quantile=float(cfg.benchmark.top_quantile))
    preds_df = trainer.predict(test_loader)
    preds_path = run_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    bt_metrics = SimpleLongShortBacktester(
        top_quantile=float(cfg.benchmark.top_quantile)
    ).evaluate(preds_df)

    # 5. Persist summary -------------------------------------------------------------
    summary: Dict[str, Any] = {
        "model": str(cfg.model.name),
        "best_epoch": fit_res["best_epoch"],
        "val_rank_ic": fit_res["best_val_rank_ic"],
        "train_time_sec": fit_res["train_time_sec"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"bt_{k}": v for k, v in bt_metrics.items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (run_dir / "history.json").write_text(json.dumps(fit_res["history"], indent=2, default=float))

    best_ckpt = ckpt_dir / f"{model.__class__.__name__.lower()}_best.pt"
    tracker.summary(summary)
    tracker.save_file(best_ckpt)
    tracker.finish()

    logger.info(f"Done. Summary: {json.dumps(summary, default=float)}")
    return {
        "summary": summary,
        "history": fit_res["history"],
        "checkpoint": str(best_ckpt),
        "predictions": str(preds_path),
        "run_dir": str(run_dir),
    }
