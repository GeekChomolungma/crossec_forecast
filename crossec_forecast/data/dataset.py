import re
from pathlib import Path
from typing import List, Optional, Set, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PanelTimeSeriesDataset(Dataset):
    """
    High-performance Time-Series Dataset for Financial Panel Data.
    Preserves (timestamp, symbol) ordering, supports per-symbol lookback window L=6,
    handles asynchronous listing dates, and filters unrevealed trailing targets.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        seq_len: int = 6,
        target_col: str = "logret1_win",
        fwd_ret_col: str = "fwd_logret_1",
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
        feature_pattern: str = r"^crossec_.*_mad_Zscore$",
        feature_cols: Optional[List[str]] = None,
        allowed_timestamps: Optional[Set[Any]] = None,
        is_inference: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.target_col = target_col
        self.fwd_ret_col = fwd_ret_col
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col
        self.is_inference = is_inference

        # Load DataFrame if path is provided
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        else:
            df = data.copy()

        # Identify feature columns
        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
            regex = re.compile(feature_pattern)
            self.feature_cols = [c for c in df.columns if regex.match(c)]

        if not self.feature_cols:
            raise ValueError(
                f"No feature columns found matching pattern '{feature_pattern}' in DataFrame."
            )

        # Ensure sorted by (timestamp, symbol)
        if not df[timestamp_col].is_monotonic_increasing:
            df = df.sort_values(by=[timestamp_col, symbol_col]).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.num_features = len(self.feature_cols)

        # Precompute feature array for O(1) slicing
        self.features_np = np.nan_to_num(
            df[self.feature_cols].to_numpy(dtype=np.float32, copy=True),
            nan=0.0
        )

        # Extract meta arrays
        timestamps_raw = df[timestamp_col].values
        symbols_raw = df[symbol_col].values

        # Membership key for the split filter. TimeSplitter yields pandas Timestamps
        # (iterating a Series), while `timestamps_raw[i]` is a numpy datetime64 for a
        # datetime column — and `np.datetime64 in {Timestamp, ...}` (or across datetime64
        # units) is False on some pandas/numpy combos, which would silently drop EVERY
        # sample. For a datetime column canonicalize both sides to int64-ns; for int/str
        # tick ids plain set membership already matches.
        if allowed_timestamps is None:
            ts_key, allowed_keys = timestamps_raw, None
        elif pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            ts_key = df[timestamp_col].to_numpy(dtype="datetime64[ns]").astype("int64")
            allowed_keys = {pd.Timestamp(t).value for t in allowed_timestamps}
        else:
            ts_key, allowed_keys = timestamps_raw, set(allowed_timestamps)

        # Target routing: `target_col` is whatever column drives training loss for this
        # task (binary win label, a raw forward return for a regression/point-forecast
        # model, a longer-horizon column, ...). `fwd_ret_col` is the realized forward
        # return used for cross-sectional Rank IC / eval and is required unless inferring.
        # Both are surfaced on every sample; a model's compute_loss picks whichever it needs.
        if not is_inference and target_col not in df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in data (columns: {list(df.columns)[:12]}...). "
                "Set data.target_col to an existing column, or use is_inference=True."
            )
        if not is_inference and fwd_ret_col not in df.columns:
            raise ValueError(
                f"fwd_ret_col '{fwd_ret_col}' not found in data — it is needed for Rank IC / eval."
            )
        target_raw = df[target_col].values if target_col in df.columns else np.full(len(df), np.nan)
        fwd_ret_raw = df[fwd_ret_col].values if fwd_ret_col in df.columns else np.full(len(df), np.nan)

        # Build sliding window indices per symbol
        self.samples: List[Dict[str, Any]] = []
        
        # Group row indices by symbol
        sym_to_indices = df.groupby(symbol_col, sort=False).indices

        for sym, idx_array in sym_to_indices.items():
            n_sym = len(idx_array)
            if n_sym < seq_len:
                # Skip symbols with insufficient history (< L steps)
                continue

            for i in range(seq_len - 1, n_sym):
                curr_row_idx = idx_array[i]
                curr_ts = timestamps_raw[curr_row_idx]

                # Check timestamp filter if provided (e.g. Train / Val / Test split)
                if allowed_keys is not None and ts_key[curr_row_idx] not in allowed_keys:
                    continue

                curr_y = target_raw[curr_row_idx]
                curr_fwd = fwd_ret_raw[curr_row_idx]

                # Training / eval: drop any sample whose target OR forward return is NaN.
                # Never zero-fill — a fake 0.0 target/return silently corrupts the loss
                # and Rank IC (this bit the old code for regression / multi-horizon targets).
                if not is_inference and (pd.isna(curr_y) or pd.isna(curr_fwd)):
                    continue

                # Window row indices: [t-L+1, ..., t]
                window_row_indices = idx_array[i - seq_len + 1 : i + 1]

                self.samples.append({
                    "indices": window_row_indices,
                    "y": float(curr_y) if not pd.isna(curr_y) else float("nan"),
                    "fwd_ret": float(curr_fwd) if not pd.isna(curr_fwd) else float("nan"),
                    "symbol": str(sym),
                    "timestamp": curr_ts,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        x_np = self.features_np[item["indices"]]  # Shape: [seq_len, num_features]
        return {
            "x": torch.from_numpy(x_np),                             # [L, D]
            "y": torch.tensor([item["y"]], dtype=torch.float32),     # [1]
            "fwd_logret": torch.tensor([item["fwd_ret"]], dtype=torch.float32),  # [1]
            "symbol": item["symbol"],
            "timestamp": item["timestamp"],
        }
