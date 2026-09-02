import math
import unittest
import torch
import torch.nn.functional as F
from crossec_forecast.configs import DataConfig, TrainConfig
from crossec_forecast.data import build_dataloaders
from crossec_forecast.models import build_model
from crossec_forecast.models.base import BaseClassifierModel
from crossec_forecast.engine import Trainer
from examples.mock_panel_data import generate_mock_panel_data


class _MeanRuleZeroShot(BaseClassifierModel):
    """Parameter-free zero-shot baseline: score = mean of the last-step features.

    Stands in for a real frozen TSFM run — exercises the Trainer eval-only path
    without needing an optional backend installed.
    """

    output_kind = "point_forecast"
    zero_shot = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x[:, -1, :].mean(dim=1, keepdim=True)  # [B, 1], no parameters

    def to_score(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.reshape(raw.shape[0])

    def compute_loss(self, raw: torch.Tensor, batch) -> torch.Tensor:
        target = batch["y"].to(raw.device).reshape(raw.shape)
        return F.huber_loss(raw, target)  # reported as val_loss only, never backpropped


class TestTrainerWorkflow(unittest.TestCase):

    def setUp(self):
        self.df = generate_mock_panel_data(
            output_path=None,
            num_timestamps=40,
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

    def test_trainer_fit_and_predict(self):
        model_cfg = {
            "seq_len": self.meta["seq_len"],
            "feature_dim": self.meta["num_features"],
            "num_classes": 1,
            "hidden_dims": [16],
            "use_norm": False,
        }
        model = build_model("mlp", model_cfg)

        train_cfg = TrainConfig(
            epochs=3,
            lr=1e-3,
            early_stopping_patience=2,
            device="cpu",
            checkpoint_dir="./test_checkpoints",
        )

        trainer = Trainer(model=model, config=train_cfg)
        fit_res = trainer.fit(self.train_loader, self.val_loader)

        self.assertIn("history", fit_res)
        self.assertIn("best_val_rank_ic", fit_res)
        self.assertGreaterEqual(fit_res["best_epoch"], 1)

        # Predict
        preds_df = trainer.predict(self.test_loader)
        self.assertIn("pred_prob", preds_df.columns)
        self.assertIn("target", preds_df.columns)
        self.assertIn("fwd_logret_1", preds_df.columns)
        self.assertEqual(len(preds_df), self.meta["n_test_samples"])

        # Evaluate
        eval_metrics = trainer.evaluate(self.test_loader)
        self.assertIn("mean_rank_ic", eval_metrics)
        self.assertIn("auc", eval_metrics)
        self.assertIn("accuracy", eval_metrics)


class TestTrainerZeroShot(unittest.TestCase):
    """Part 1: Trainer eval-only lifecycle for parameter-free / zero_shot models."""

    def setUp(self):
        self.df = generate_mock_panel_data(output_path=None, num_timestamps=40, seed=7)
        self.data_config = DataConfig(
            seq_len=6, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            embargo_steps=1, batch_size=32,
        )
        self.train_loader, self.val_loader, self.test_loader, self.meta = build_dataloaders(
            self.df, self.data_config
        )
        self.model_cfg = {
            "seq_len": self.meta["seq_len"],
            "feature_dim": self.meta["num_features"],
            "num_classes": 1,
        }
        self.train_cfg = TrainConfig(
            epochs=5, early_stopping_patience=2, device="cpu",
            checkpoint_dir="./test_zeroshot_checkpoints",
        )

    def _trainer(self):
        return Trainer(model=_MeanRuleZeroShot(self.model_cfg), config=self.train_cfg)

    def test_eval_only_detected_no_optimizer(self):
        trainer = self._trainer()
        self.assertTrue(trainer.eval_only)
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(trainer.scheduler)

    def test_fit_scores_validation_once(self):
        trainer = self._trainer()
        fit_res = trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(len(fit_res["history"]), 1)
        self.assertEqual(fit_res["best_epoch"], 0)
        self.assertTrue(math.isfinite(fit_res["best_val_rank_ic"]))
        self.assertEqual(
            fit_res["history"][0]["val_mean_rank_ic"], fit_res["best_val_rank_ic"]
        )
        # checkpoint persisted so run_train / benchmark / infer stay uniform
        self.assertTrue(trainer.best_checkpoint_path.exists())

    def test_predict_and_evaluate_still_work(self):
        trainer = self._trainer()
        trainer.fit(self.train_loader, self.val_loader)

        preds_df = trainer.predict(self.test_loader)
        self.assertIn("pred_prob", preds_df.columns)
        self.assertEqual(len(preds_df), self.meta["n_test_samples"])

        metrics = trainer.evaluate(self.test_loader)
        self.assertIn("mean_rank_ic", metrics)
        # point_forecast -> classification block is gated off
        self.assertNotIn("auc", metrics)

    def test_train_epoch_rejected_on_eval_only(self):
        trainer = self._trainer()
        with self.assertRaises(AssertionError):
            trainer.train_epoch(self.train_loader)

    def test_trained_model_keeps_training_path(self):
        model = build_model("mlp", {**self.model_cfg, "hidden_dims": [8], "use_norm": False})
        trainer = Trainer(model=model, config=self.train_cfg)
        self.assertFalse(trainer.eval_only)
        self.assertIsNotNone(trainer.optimizer)


if __name__ == "__main__":
    unittest.main()

