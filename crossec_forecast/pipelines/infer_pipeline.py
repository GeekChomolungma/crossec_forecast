"""Batch inference with a trained checkpoint over a full panel (no train/val/test split)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..data.dataloader import panel_collate_fn
from ..data.dataset import PanelTimeSeriesDataset
from ..models import build_model  # noqa: F401  (also populates the registry)
from ..utils.logger import setup_logger
from .context import model_build_config, select_device


def run_infer(
    cfg: DictConfig,
    run_dir: Path,
    checkpoint: str | Path,
    *,
    data_path: Optional[str] = None,
    output_name: str = "predictions.csv",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Score every revealed (symbol, timestamp) row of a panel.

    Unlike training, this builds ONE inference dataset (``is_inference=True``,
    no timestamp filtering) so the whole file is scored in chronological order.
    """
    logger = logger or setup_logger("pipeline.infer")
    run_dir = Path(run_dir)
    path = data_path or str(cfg.data.path)
    feature_cols = (
        list(cfg.data.feature_cols) if cfg.data.feature_cols is not None else None
    )
    cov_cols = (
        list(cfg.data.cov_cols) if cfg.data.cov_cols is not None else None
    )

    dataset = PanelTimeSeriesDataset(
        data=path,
        seq_len=int(cfg.data.seq_len),
        target_col=str(cfg.data.target_col),
        fwd_ret_col=str(cfg.data.fwd_ret_col),
        timestamp_col=str(cfg.data.timestamp_col),
        symbol_col=str(cfg.data.symbol_col),
        feature_pattern=str(cfg.data.feature_pattern),
        feature_cols=feature_cols,
        cov_pattern=(str(cfg.data.cov_pattern) if cfg.data.cov_pattern is not None else None),
        cov_cols=cov_cols,
        is_inference=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        collate_fn=panel_collate_fn,
        num_workers=int(cfg.data.num_workers),
    )

    model_cfg = model_build_config(
        cfg.model,
        seq_len=dataset.seq_len,
        feature_dim=dataset.num_features,
        cov_dim=dataset.num_cov,
    )
    model = build_model(str(cfg.model.name), model_cfg)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    device = select_device(str(cfg.train.device))
    model.to(device)
    logger.info(f"Loaded {cfg.model.name} from {checkpoint} | rows to score={len(dataset)} | device={device}")

    records = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            raw = model(x)
            # `pred_prob` column name kept for back-compat; holds to_score() output
            # (a probability for binary_prob models, a raw score otherwise). See TO_IMPROVE.md C9.
            scores = model.to_score(raw).detach().cpu().numpy().reshape(-1)
            fwd = batch["fwd_logret"].squeeze(-1).numpy()
            for sym, ts, p, r in zip(batch["symbols"], batch["timestamps"], scores, fwd):
                records.append(
                    {"timestamp": ts, "symbol": sym, "pred_prob": float(p), "fwd_logret_1": float(r)}
                )

    out_df = pd.DataFrame.from_records(records).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    out_path = run_dir / output_name
    out_df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out_df)} predictions -> {out_path}")

    return {"predictions": str(out_path), "n_rows": int(len(out_df))}
