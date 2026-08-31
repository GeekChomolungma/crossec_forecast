import shutil
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from crossec_forecast.pipelines import (
    load_config,
    load_experiment,
    run_infer,
    run_train,
    to_data_config,
    to_train_config,
)
from examples.mock_panel_data import generate_mock_panel_data

CONFIG = Path("experiments/experiment.yaml")


class TestExperimentPipelines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path("./test_pipeline_tmp")
        cls.tmp.mkdir(exist_ok=True)
        cls.data_path = cls.tmp / "panel.csv"
        generate_mock_panel_data(output_path=str(cls.data_path), num_timestamps=40, seed=7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree("./runs", ignore_errors=True)

    def _overrides(self, **extra):
        base = [
            f"data.path={self.data_path.as_posix()}",
            f"run.output_root={(self.tmp / 'runs').as_posix()}",
            "train.epochs=2",
            "train.device=cpu",
            "wandb.enabled=false",
        ]
        base += [f"{k}={v}" for k, v in extra.items()]
        return base

    def test_config_merge_precedence(self):
        cfg = load_config(CONFIG, ["model.name=dlinear", "train.lr=0.002"])
        self.assertEqual(cfg.model.name, "dlinear")      # CLI override
        self.assertEqual(cfg.train.lr, 0.002)            # CLI override
        self.assertEqual(cfg.data.seq_len, 6)            # from yaml
        self.assertEqual(cfg.experiment.seed, 42)        # from schema default via yaml

    def test_config_rejects_unknown_key(self):
        with self.assertRaises(Exception):
            load_config(CONFIG, ["trian.lr=0.1"])        # typo -> struct error

    def test_adapters_roundtrip(self):
        cfg = load_config(CONFIG, self._overrides())
        dc = to_data_config(cfg)
        tc = to_train_config(cfg, "./ckpt")
        self.assertEqual(dc.seq_len, 6)
        self.assertEqual(tc.epochs, 2)
        self.assertEqual(tc.checkpoint_dir, "./ckpt")

    def test_run_train_then_infer(self):
        cfg, run_dir, logger = load_experiment(
            CONFIG, self._overrides(**{"model.name": "mlp", "run.name": "pt_mlp"}), job_type="train"
        )
        out = run_train(cfg, run_dir, logger)

        self.assertTrue(Path(out["checkpoint"]).exists())
        self.assertTrue(Path(out["predictions"]).exists())
        self.assertTrue((run_dir / "config.yaml").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertIn("test_auc", out["summary"])

        infer_out = run_infer(cfg, run_dir, out["checkpoint"], output_name="infer.csv", logger=logger)
        self.assertTrue(Path(infer_out["predictions"]).exists())
        self.assertGreater(infer_out["n_rows"], 0)


if __name__ == "__main__":
    unittest.main()
