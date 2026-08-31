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
                if allowed_timestamps is not None and curr_ts not in allowed_timestamps:
                    continue

                curr_y = target_raw[curr_row_idx]

                # If training/evaluating, skip unrevealed target (NaN)
                if not is_inference and pd.isna(curr_y):
                    continue

                curr_fwd = fwd_ret_raw[curr_row_idx]
                if pd.isna(curr_fwd):
                    curr_fwd = 0.0

                # Window row indices: [t-L+1, ..., t]
                window_row_indices = idx_array[i - seq_len + 1 : i + 1]

                self.samples.append({
                    "indices": window_row_indices,
                    "y": float(curr_y) if not pd.isna(curr_y) else 0.0,
                    "fwd_ret": float(curr_fwd),
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
