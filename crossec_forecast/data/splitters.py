import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Union


class TimeSplitter:
    """
    Chronological time series splitter with embargo gap support.
    Ensures zero temporal leakage between Train, Validation, and Test partitions.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        embargo_steps: int = 1,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.embargo_steps = embargo_steps

    def split_timestamps(
        self, timestamps: Union[List[Any], pd.Series, np.ndarray]
    ) -> Tuple[Set[Any], Set[Any], Set[Any]]:
        """
        Split a sequence of unique sorted timestamps into Train, Val, Test sets
        with embargo steps purged after the train and val cutoffs.
        """
        unique_ts = sorted(list(set(timestamps)))
        n_total = len(unique_ts)
        if n_total < 3:
            raise ValueError(f"Need at least 3 unique timestamps to split, got {n_total}")

        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        # Train range
        train_ts = unique_ts[:n_train]

        # Embargo after train
        val_start = min(n_train + self.embargo_steps, n_total)
        val_end = min(val_start + n_val, n_total)
        val_ts = unique_ts[val_start:val_end]

        # Embargo after val
        test_start = min(val_end + self.embargo_steps, n_total)
        test_ts = unique_ts[test_start:]

        return set(train_ts), set(val_ts), set(test_ts)

