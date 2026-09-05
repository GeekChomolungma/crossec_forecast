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

    Four independent column-selection knobs, each resolved here (pattern-or-explicit,
    explicit wins) into a concrete list + count surfaced as ``self.<name>_cols`` /
    ``self.num_<name>`` and, downstream, auto-injected into every model's config
    (``pipelines.context.model_build_config``) as ``self.<name>_dim`` on
    ``models.base.BaseClassifierModel`` — the model reads its own shape off itself, it
    never sees raw column names:

      * ``feature_cols`` / ``feature_pattern`` — the primary model input (typically the
        crossec_* z-scored panel).
      * ``cov_cols`` / ``cov_pattern`` — auxiliary numeric channels, opt-in.
      * ``extra_input_cols`` / ``extra_input_pattern`` — RAW/passthrough columns for a
        model's own internal use (e.g. a raw price series a TSFM wrapper log-diffs
        itself), opt-in. Kept distinct from cov_cols so "feed me this raw column
        verbatim" doesn't overload feature_cols' established meaning elsewhere.
      * ``target_cols`` — explicit list only (no pattern variant), drives ``batch["y"]``
        (a model's own loss target(s), read by ``compute_loss``). Defaults to a single
        column so existing single-target models are unaffected.

    ``x`` packing convention: ``x`` is a single ``[seq_len, feature_dim + cov_dim +
    extra_input_dim]`` block — feature columns, then covariate columns, then extra-input
    columns, in that fixed order. There is no separate channel threaded through the
    Trainer for any of these: a model recovers its own slice from ``x`` using
    ``self.feature_dim`` / ``self.cov_dim`` / ``self.extra_input_dim``. At least one of
    the three groups must resolve non-empty (``feature_cols`` alone is no longer required
    to be non-empty — a model that only needs ``extra_input_cols`` can set
    ``feature_cols: []``).

    ``fwd_ret_col`` is **not** one of the four column groups above and is never
    model-configurable per-target: it is the single, shared, mandatory Rank-IC ground
    truth every model in a run is scored against (``batch["fwd_logret"]``), independent
    of ``target_cols``. Folding it into ``target_cols`` would let different models in the
    same benchmark rank against different truths, breaking cross-model comparability.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        seq_len: int = 6,
        target_cols: Optional[List[str]] = None,
        fwd_ret_col: str = "fwd_logret_1",
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
        feature_pattern: str = r"^crossec_.*_mad_Zscore$",
        feature_cols: Optional[List[str]] = None,
        cov_pattern: Optional[str] = None,
        cov_cols: Optional[List[str]] = None,
        extra_input_pattern: Optional[str] = None,
        extra_input_cols: Optional[List[str]] = None,
        allowed_timestamps: Optional[Set[Any]] = None,
        is_inference: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.target_cols = list(target_cols) if target_cols else ["logret1_win"]
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

        # -- resolve the three `x`-packing column groups (pattern-or-explicit, explicit wins) --
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
        else:
            regex = re.compile(feature_pattern)
            self.feature_cols = [c for c in df.columns if regex.match(c)]

        if cov_cols is not None:
            self.cov_cols = list(cov_cols)
        elif cov_pattern is not None:
            cov_regex = re.compile(cov_pattern)
            self.cov_cols = [c for c in df.columns if cov_regex.match(c)]
        else:
            self.cov_cols = []

        if extra_input_cols is not None:
            self.extra_input_cols = list(extra_input_cols)
        elif extra_input_pattern is not None:
            extra_regex = re.compile(extra_input_pattern)
            self.extra_input_cols = [c for c in df.columns if extra_regex.match(c)]
        else:
            self.extra_input_cols = []

        if not (self.feature_cols or self.cov_cols or self.extra_input_cols):
            raise ValueError(
                "No input columns resolved: feature_cols/feature_pattern, cov_cols/"
                "cov_pattern and extra_input_cols/extra_input_pattern are all empty. "
                "At least one of the three must select at least one column."
            )

        # Ensure sorted by (timestamp, symbol)
        if not df[timestamp_col].is_monotonic_increasing:
            df = df.sort_values(by=[timestamp_col, symbol_col]).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.num_features = len(self.feature_cols)
        self.num_cov = len(self.cov_cols)
        self.num_extra_input = len(self.extra_input_cols)
        self.num_target = len(self.target_cols)

        # Precompute the packed [feature..., cov..., extra_input...] array once for O(1)
        # slicing. Column order is fixed — see the class docstring for the `x` packing
        # convention every model relies on via self.feature_dim / cov_dim / extra_input_dim.
        packed_cols = self.feature_cols + self.cov_cols + self.extra_input_cols
        self.packed_np = np.nan_to_num(
            df[packed_cols].to_numpy(dtype=np.float32, copy=True),
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

        # Target routing: `target_cols` drive batch["y"] — whatever a model's own
        # compute_loss needs (binary win label, a raw forward return for a
        # regression/point-forecast model, several columns for a multi-target model, ...).
        # `fwd_ret_col` is the realized forward return used for cross-sectional Rank IC /
        # eval and is required unless inferring — it is NOT part of target_cols (see class
        # docstring). Both are surfaced on every sample.
        if not is_inference:
            missing = [c for c in self.target_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"target_cols {missing} not found in data (columns: {list(df.columns)[:12]}...). "
                    "Set data.target_cols to existing columns, or use is_inference=True."
                )
        if not is_inference and fwd_ret_col not in df.columns:
            raise ValueError(
                f"fwd_ret_col '{fwd_ret_col}' not found in data — it is needed for Rank IC / eval."
            )
        target_cols_present = [c for c in self.target_cols if c in df.columns]
        if target_cols_present:
            target_raw = df[target_cols_present].to_numpy(dtype=np.float64)
            if len(target_cols_present) != len(self.target_cols):
                # is_inference path with some target_cols missing (e.g. scoring a fresh
                # panel that never had labels): pad the missing ones with NaN columns so
                # target_raw stays aligned to self.target_cols positionally.
                full = np.full((len(df), self.num_target), np.nan, dtype=np.float64)
                present_idx = [self.target_cols.index(c) for c in target_cols_present]
                full[:, present_idx] = target_raw
                target_raw = full
        else:
            target_raw = np.full((len(df), self.num_target), np.nan, dtype=np.float64)
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

                curr_y = target_raw[curr_row_idx]  # [num_target]
                curr_fwd = fwd_ret_raw[curr_row_idx]

                # Training / eval: drop any sample where ANY target_cols value or the
                # forward return is NaN. Never zero-fill — a fake 0.0 target/return
                # silently corrupts the loss and Rank IC (this bit the old code for
                # regression / multi-horizon targets).
                if not is_inference and (np.isnan(curr_y).any() or pd.isna(curr_fwd)):
                    continue

                # Window row indices: [t-L+1, ..., t]
                window_row_indices = idx_array[i - seq_len + 1 : i + 1]

                self.samples.append({
                    "indices": window_row_indices,
                    "y": curr_y.copy(),
                    "fwd_ret": float(curr_fwd) if not pd.isna(curr_fwd) else float("nan"),
                    "symbol": str(sym),
                    "timestamp": curr_ts,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        x_np = self.packed_np[item["indices"]]  # [seq_len, feature_dim+cov_dim+extra_input_dim]
        return {
            "x": torch.from_numpy(x_np),                                    # [L, D]
            "y": torch.tensor(item["y"], dtype=torch.float32),              # [num_target]
            "fwd_logret": torch.tensor([item["fwd_ret"]], dtype=torch.float32),  # [1]
            "symbol": item["symbol"],
            "timestamp": item["timestamp"],
        }
