import math
import os
import unittest
import shutil
from pathlib import Path
from unittest import mock
from crossec_forecast.configs import DataConfig, TrainConfig, BenchmarkConfig
from crossec_forecast.data import build_dataloaders
from crossec_forecast.eval import BenchmarkEngine
from examples.mock_panel_data import generate_mock_panel_data


class TestBenchmarkEngine(unittest.TestCase):

    def setUp(self):
        self.export_dir = Path("./test_benchmark_reports")
        self.df = generate_mock_panel_data(
            output_path=None,
            num_timestamps=35,
            seed=42,
        )
        self.data_config = DataConfig(
            seq_len=6,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_steps=1,
            batch_size=32,
        )
        self.train_loader, self.val_loader, self.test_loader, self.meta = build_dataloaders(
            self.df, self.data_config
        )

    def tearDown(self):
        if self.export_dir.exists():
            shutil.rmtree(self.export_dir, ignore_errors=True)
        ckpt_dir = Path("./test_bench_ckpts")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def _engine(self):
        train_cfg = TrainConfig(
            epochs=2,
            lr=1e-3,
            device="cpu",
            checkpoint_dir="./test_bench_ckpts",
        )
        bench_cfg = BenchmarkConfig(
            models=[
                {"name": "mlp", "config": {"hidden_dims": [16], "use_norm": False}},
                {"name": "dlinear", "config": {"individual": False}},
            ],
            export_dir=str(self.export_dir),
        )
        return BenchmarkEngine(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            meta_info=self.meta,
            train_config=train_cfg,
            benchmark_config=bench_cfg,
        )

    @mock.patch.dict(os.environ, {"CF_BENCH_RUN_TEST": "1"})
    def test_benchmark_run_multi_model_with_oos(self):
        res_df = self._engine().run()
        self.assertEqual(len(res_df), 2)
        for col in ("model", "val_rank_ic", "test_rank_ic", "test_auc"):
            self.assertIn(col, res_df.columns)
        # OOS test ran -> test / backtest columns are populated
        self.assertTrue(res_df["test_rank_ic"].notna().all())
        self.assertTrue(res_df["sharpe_ratio"].notna().all())

        self.assertTrue((self.export_dir / "benchmark_summary.csv").exists())
        self.assertTrue((self.export_dir / "benchmark_summary.md").exists())
        self.assertTrue((self.export_dir / "benchmark_summary.json").exists())

    @mock.patch.dict(os.environ, {"CF_BENCH_RUN_TEST": ""})
    def test_benchmark_default_is_validation_only(self):
        res_df = self._engine().run()
        self.assertEqual(len(res_df), 2)
        # same column schema, but the OOS test was skipped -> test / backtest are NaN
        self.assertIn("test_rank_ic", res_df.columns)
        self.assertTrue(res_df["test_rank_ic"].isna().all())
        self.assertTrue(res_df["ann_return"].isna().all())
        self.assertTrue(res_df["val_rank_ic"].map(math.isfinite).all())
        # sorted by val_rank_ic descending
        self.assertTrue(res_df["val_rank_ic"].is_monotonic_decreasing)
        self.assertTrue((self.export_dir / "benchmark_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()

