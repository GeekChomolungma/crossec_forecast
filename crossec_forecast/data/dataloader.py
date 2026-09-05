from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PanelTimeSeriesDataset
from .splitters import TimeSplitter
from ..configs.default_config import DataConfig


def panel_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for financial panel time-series batches."""
    x = torch.stack([item["x"] for item in batch], dim=0)               # [B, L, D]
    y = torch.stack([item["y"] for item in batch], dim=0)               # [B, 1]
    fwd_ret = torch.stack([item["fwd_logret"] for item in batch], dim=0) # [B, 1]
    symbols = [item["symbol"] for item in batch]
    timestamps = [item["timestamp"] for item in batch]

    return {
        "x": x,
        "y": y,
        "fwd_logret": fwd_ret,
        "symbols": symbols,
        "timestamps": timestamps,
    }


def build_dataloaders(
    data: Union[pd.DataFrame, str, Path],
    config: Optional[DataConfig] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Build standardized Train, Validation, and Test DataLoaders with zero temporal leakage.
    """
    if config is None:
        config = DataConfig(**kwargs)

    # Load dataframe to obtain unique timestamps for splitting
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    else:
        df = data

    if config.timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{config.timestamp_col}' not found in data.")

    # Chronological Split
    splitter = TimeSplitter(
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        embargo_steps=config.embargo_steps,
    )
    train_ts, val_ts, test_ts = splitter.split_timestamps(df[config.timestamp_col])

    # Datasets. `cov_cols` / `extra_input_cols` are resolved from train (pattern-matched or
    # explicit) and then pinned explicitly for val/test — same reasoning as feature_cols:
    # all three splits must pack `x` with the identical column set/order. `target_cols` has
    # no pattern variant, so it is just passed through identically to all three.
    train_dataset = PanelTimeSeriesDataset(
        data=df,
        seq_len=config.seq_len,
        target_cols=config.target_cols,
        fwd_ret_col=config.fwd_ret_col,
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
        feature_pattern=config.feature_pattern,
        feature_cols=config.feature_cols,
        cov_pattern=config.cov_pattern,
        cov_cols=config.cov_cols,
        extra_input_pattern=config.extra_input_pattern,
        extra_input_cols=config.extra_input_cols,
        allowed_timestamps=train_ts,
    )

    feature_cols = train_dataset.feature_cols
    cov_cols = train_dataset.cov_cols
    extra_input_cols = train_dataset.extra_input_cols

    val_dataset = PanelTimeSeriesDataset(
        data=df,
        seq_len=config.seq_len,
        target_cols=config.target_cols,
        fwd_ret_col=config.fwd_ret_col,
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
        feature_cols=feature_cols,
        cov_cols=cov_cols,
        extra_input_cols=extra_input_cols,
        allowed_timestamps=val_ts,
    )

    test_dataset = PanelTimeSeriesDataset(
        data=df,
        seq_len=config.seq_len,
        target_cols=config.target_cols,
        fwd_ret_col=config.fwd_ret_col,
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
        feature_cols=feature_cols,
        cov_cols=cov_cols,
        extra_input_cols=extra_input_cols,
        allowed_timestamps=test_ts,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        collate_fn=panel_collate_fn,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=panel_collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=panel_collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
    )

    meta_info = {
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "cov_cols": cov_cols,
        "num_cov": len(cov_cols),
        "extra_input_cols": extra_input_cols,
        "num_extra_input": len(extra_input_cols),
        "target_cols": config.target_cols,
        "num_target": train_dataset.num_target,
        "seq_len": config.seq_len,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "n_test_samples": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, meta_info

