import unittest
import torch
from crossec_forecast.configs import DataConfig, TrainConfig
from crossec_forecast.data import build_dataloaders
from crossec_forecast.models import build_model
from crossec_forecast.engine import Trainer
from examples.mock_panel_data import generate_mock_panel_data


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


if __name__ == "__main__":
    unittest.main()

