from pathlib import Path
from typing import Dict, Any, Optional
import time
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from ..configs.default_config import TrainConfig
from ..models.base import BaseClassifierModel
from ..eval.metrics import compute_cross_sectional_rank_ic, compute_all_metrics
from ..utils.logger import setup_logger


class Trainer:
    """
    Unified Trainer implementing the Three-Stage Quant Lifecycle:
    1. Train: Optimizes continuous loss & backpropagates gradients.
    2. Validation: Evaluates cross-sectional Mean Rank IC, saves Best Quant Model, and drives Early Stopping.
    3. Test: Evaluates generalized OOS performance.
    """

    def __init__(
        self,
        model: BaseClassifierModel,
        config: Optional[TrainConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        logger=None,
        callbacks: Optional[list] = None,
    ):
        self.config = config or TrainConfig()
        self.logger = logger or setup_logger("Trainer")
        # Optional list of callables invoked as cb(epoch_log: dict) after every epoch.
        self.callbacks = list(callbacks) if callbacks else []
        self.device = self._select_device(self.config.device)
        self.model = model.to(self.device)

        # Loss is owned by the model (model.compute_loss) — the Trainer is loss-agnostic.

        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        elif self.config.scheduler_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=self.config.scheduler_patience, # 连续几轮不创新高就降LR
                factor=self.config.scheduler_factor,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None

        # Checkpoints
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / f"{model.__class__.__name__.lower()}_best.pt"

    def _select_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def train_epoch(self, train_loader) -> float:
        """Stage 1: Gradient Backpropagation and Feature Space Shaping."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)

            self.optimizer.zero_grad()
            raw = self.model(x)
            loss = self.model.compute_loss(raw, batch)
            loss.backward()

            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Stage 2: Validation Business Metric (Cross-Sectional Rank IC) Evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_scores = []
        all_returns = []
        all_dates = []

        for batch in val_loader:
            x = batch["x"].to(self.device)

            raw = self.model(x)
            loss = self.model.compute_loss(raw, batch)
            total_loss += loss.item()
            num_batches += 1

            scores = self.model.to_score(raw).detach().cpu().numpy().reshape(-1)
            returns = batch["fwd_logret"].squeeze(-1).numpy()
            dates = batch["timestamps"]

            all_scores.extend(scores)
            all_returns.extend(returns)
            all_dates.extend(dates)

        avg_loss = total_loss / max(1, num_batches)
        mean_rank_ic, ic_ir, _ = compute_cross_sectional_rank_ic(all_scores, all_returns, all_dates)

        return {
            "val_loss": avg_loss,
            "val_mean_rank_ic": mean_rank_ic,
            "val_ic_ir": ic_ir,
        }

    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        """Execute full training and validation loop with Rank IC early stopping."""
        best_val_ic = -float("inf")
        best_epoch = 0
        patience_counter = 0
        history = []

        start_time = time.time()
        self.logger.info(f"Starting training on device: {self.device}")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_ic = val_metrics["val_mean_rank_ic"]
            val_loss = val_metrics["val_loss"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_ic)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_rank_ic": val_ic,
                "val_ic_ir": val_metrics["val_ic_ir"],
                "lr": current_lr,
            }
            history.append(epoch_log)

            # Checkpoint selection strictly driven by Validation Rank IC. `min_delta` requires
            # a gain of at least that much to count as improvement, so noise-level IC wobble
            # doesn't reset the early-stopping patience counter.
            if val_ic > best_val_ic + self.config.min_delta:
                best_val_ic = val_ic
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                patience_counter = 0
                is_best = " [BEST IC SAVED]"
            else:
                patience_counter += 1
                is_best = ""

            # Fire epoch callbacks (e.g. experiment trackers). Never let them break training.
            epoch_log["is_best"] = bool(is_best)
            for cb in self.callbacks:
                try:
                    cb(epoch_log)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"Epoch callback {cb!r} failed: {exc}")

            self.logger.info(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Rank IC: {val_ic:.4f} | ICIR: {val_metrics['val_ic_ir']:.2f}{is_best}"
            )

            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, best Rank IC: {best_val_ic:.4f})"
                )
                break

        elapsed_time = time.time() - start_time

        # Reload best weights for subsequent inference & testing
        if self.best_checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=True))
            self.logger.info(f"Loaded best model checkpoint from {self.best_checkpoint_path}")

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_rank_ic": best_val_ic,
            "train_time_sec": elapsed_time,
        }

    @torch.no_grad()
    def predict(self, loader) -> pd.DataFrame:
        """Generate full predictions DataFrame for testing or live inference."""
        self.model.eval()
        all_scores = []
        all_targets = []
        all_returns = []
        all_symbols = []
        all_dates = []

        for batch in loader:
            x = batch["x"].to(self.device)
            raw = self.model(x)
            scores = self.model.to_score(raw).detach().cpu().numpy().reshape(-1)
            targets = batch["y"].squeeze(-1).numpy()
            returns = batch["fwd_logret"].squeeze(-1).numpy()

            all_scores.extend(scores)
            all_targets.extend(targets)
            all_returns.extend(returns)
            all_symbols.extend(batch["symbols"])
            all_dates.extend(batch["timestamps"])

        # Column name kept as `pred_prob` for artifact/back-compat reasons; for
        # non-"binary_prob" models this holds the raw `to_score` value (see TO_IMPROVE.md C9).
        return pd.DataFrame({
            "timestamp": all_dates,
            "symbol": all_symbols,
            "pred_prob": all_scores,
            "target": all_targets,
            "fwd_logret_1": all_returns,
        })

    def evaluate(self, test_loader, top_quantile: float = 0.2) -> Dict[str, float]:
        """Stage 3: Comprehensive Out-Of-Sample (Test) Evaluation."""
        preds_df = self.predict(test_loader)
        metrics = compute_all_metrics(
            preds=preds_df["pred_prob"].values,
            targets=preds_df["target"].values,
            returns=preds_df["fwd_logret_1"].values,
            dates=preds_df["timestamp"].values,
            top_quantile=top_quantile,
            output_kind=getattr(self.model, "output_kind", "binary_prob"),
        )
        return metrics

