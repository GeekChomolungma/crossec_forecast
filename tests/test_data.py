import unittest
import numpy as np
import pandas as pd
import torch

from crossec_forecast.data.splitters import TimeSplitter
from crossec_forecast.data.dataset import PanelTimeSeriesDataset
from crossec_forecast.data.dataloader import build_dataloaders
from crossec_forecast.configs.default_config import DataConfig
from examples.mock_panel_data import generate_mock_panel_data


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.df = generate_mock_panel_data(
            output_path=None,
            num_timestamps=50,
            seed=42,
        )

    def test_time_splitter_embargo(self):
        timestamps = list(range(100))
        splitter = TimeSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, embargo_steps=2)
        train_ts, val_ts, test_ts = splitter.split_timestamps(timestamps)

        self.assertEqual(len(train_ts), 70)
        self.assertEqual(len(val_ts), 15)
        # 100 - 70 - 2 (embargo) = 28; val takes 15; 28 - 15 - 2 (embargo) = 11
        self.assertEqual(len(test_ts), 11)

        # Ensure no overlap
        self.assertEqual(len(train_ts.intersection(val_ts)), 0)
        self.assertEqual(len(val_ts.intersection(test_ts)), 0)
        self.assertEqual(len(train_ts.intersection(test_ts)), 0)

        # Ensure embargo gaps
        max_train = max(train_ts)
        min_val = min(val_ts)
        self.assertEqual(min_val - max_train, 3)  # 2 embargo steps + 1

    def test_dataset_window_and_nan_filtering(self):
        dataset = PanelTimeSeriesDataset(
            data=self.df,
            seq_len=6,
            target_col="logret1_win",
            fwd_ret_col="fwd_logret_1",
            timestamp_col="timestamp",
            symbol_col="symbol",
            feature_pattern=r"^crossec_.*_mad_Zscore$",
        )

        self.assertGreater(len(dataset), 0)
        sample = dataset[0]

        # Check shapes
        self.assertEqual(sample["x"].shape, (6, dataset.num_features))
        self.assertEqual(sample["y"].shape, (1,))
        self.assertEqual(sample["fwd_logret"].shape, (1,))
        self.assertIsInstance(sample["symbol"], str)

        # Check that no sample has NaN target in training dataset
        for i in range(min(50, len(dataset))):
            s = dataset[i]
            self.assertFalse(torch.isnan(s["y"]).any())

    def test_build_dataloaders(self):
        config = DataConfig(
            seq_len=6,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_steps=1,
            batch_size=16,
        )
        train_l, val_l, test_l, meta = build_dataloaders(self.df, config)

        self.assertGreater(len(train_l), 0)
        self.assertGreater(len(val_l), 0)
        self.assertGreater(len(test_l), 0)
        self.assertEqual(meta["seq_len"], 6)
        self.assertGreater(meta["num_features"], 0)

        batch = next(iter(train_l))
        self.assertEqual(batch["x"].ndim, 3)
        self.assertEqual(batch["x"].shape[1], 6)
        self.assertEqual(batch["x"].shape[2], meta["num_features"])
        self.assertEqual(batch["y"].shape, (batch["x"].shape[0], 1))
        self.assertEqual(len(batch["symbols"]), batch["x"].shape[0])
        self.assertEqual(len(batch["timestamps"]), batch["x"].shape[0])


if __name__ == "__main__":
    unittest.main()

