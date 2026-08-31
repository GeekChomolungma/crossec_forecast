"""
Thin wandb wrapper used by every pipeline.

`wandb` is a hard dependency, but a run can still opt out via `wandb.enabled: false`
or `wandb.mode: disabled` in the experiment config, in which case every method is a
no-op and no network / disk run is created.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import wandb
from omegaconf import DictConfig, OmegaConf


class WandbTracker:
    def __init__(self, cfg: DictConfig, run_dir: Path, *, job_type: Optional[str] = None):
        w = cfg.wandb
        self.log_freq = int(w.log_freq)
        self.enabled = bool(w.enabled) and str(w.mode) != "disabled"
        self._run = None
        if not self.enabled:
            return

        init_kwargs = dict(
            project=str(w.project),
            entity=(None if w.entity is None else str(w.entity)),
            group=str(w.group) if w.group else str(cfg.experiment.name),
            job_type=str(job_type or w.job_type),
            name=str(w.name) if w.name else run_dir.name,
            notes=(None if w.notes is None else str(w.notes)),
            tags=list(w.tags) + list(cfg.run.tags),
            dir=str(run_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        # Only force `mode` when the config asks for something other than the default
        # "online"; otherwise let wandb honour the WANDB_MODE env var (offline compute
        # nodes set it in their job scripts).
        if str(w.mode) != "online":
            init_kwargs["mode"] = str(w.mode)
        self._run = wandb.init(**init_kwargs)

    # -- logging -----------------------------------------------------------------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def summary(self, metrics: Dict[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(metrics)

    def save_file(self, path: str | Path) -> None:
        if self._run is not None:
            p = Path(path)
            wandb.save(str(p), base_path=str(p.parent))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    # -- Trainer integration ---------------------------------------------------------
    def epoch_callback(self):
        """Return a ``cb(epoch_log: dict)`` suitable for ``Trainer(callbacks=[...])``."""
        def _cb(epoch_log: Dict[str, Any]) -> None:
            if not self.enabled:
                return
            epoch = int(epoch_log.get("epoch", 0))
            if self.log_freq > 0 and epoch % self.log_freq != 0:
                return
            self.log(
                {
                    "train/loss": epoch_log.get("train_loss"),
                    "val/loss": epoch_log.get("val_loss"),
                    "val/mean_rank_ic": epoch_log.get("val_mean_rank_ic"),
                    "val/ic_ir": epoch_log.get("val_ic_ir"),
                    "lr": epoch_log.get("lr"),
                    "epoch": epoch,
                },
                step=epoch,
            )

        return _cb
