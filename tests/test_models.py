import unittest
import torch
from crossec_forecast.models import (
    BaseClassifierModel,
    PretrainedBackboneModel,
    register_model,
    build_model,
    list_registered_models,
    is_model_registered,
)


class TestModelPluginSystem(unittest.TestCase):

    def setUp(self):
        self.b = 8
        self.l = 6
        self.d = 24
        self.dummy_input = torch.randn(self.b, self.l, self.d)
        self.base_cfg = {
            "seq_len": self.l,
            "feature_dim": self.d,
            "num_classes": 1,
        }

    def test_registered_models_exist(self):
        registered = list_registered_models()
        self.assertIn("mlp", registered)
        self.assertIn("lstm", registered)
        self.assertIn("dlinear", registered)

    def test_all_registered_models_forward_and_predict_proba(self):
        for name in ["mlp", "lstm", "dlinear"]:
            model = build_model(name, self.base_cfg)
            logits = model(self.dummy_input)
            self.assertEqual(logits.shape, (self.b, 1), f"Model {name} output shape mismatch")

            probs = model.predict_proba(self.dummy_input)
            self.assertEqual(probs.shape, (self.b, 1), f"Model {name} proba shape mismatch")
            self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))

    def test_all_registered_models_to_score_and_compute_loss(self):
        for name in ["mlp", "lstm", "dlinear"]:
            model = build_model(name, self.base_cfg)
            self.assertEqual(model.output_kind, "binary_prob")

            raw = model(self.dummy_input)
            score = model.to_score(raw)
            self.assertEqual(score.shape, (self.b,), f"Model {name} to_score shape mismatch")
            self.assertTrue(torch.all((score >= 0.0) & (score <= 1.0)))

            batch = {"y": torch.randint(0, 2, (self.b, 1)).float()}
            loss = model.compute_loss(raw, batch)
            self.assertEqual(loss.dim(), 0)
            loss.backward()

    def test_custom_plugin_registration(self):
        @register_model("custom_dummy_net")
        class CustomDummyNet(BaseClassifierModel):
            def __init__(self, config):
                super().__init__(config)
                self.linear = torch.nn.Linear(self.seq_len * self.feature_dim, 1)

            def forward(self, x, **kwargs):
                return self.linear(x.reshape(x.size(0), -1))

        self.assertTrue(is_model_registered("custom_dummy_net"))
        custom_model = build_model("custom_dummy_net", self.base_cfg)
        out = custom_model(self.dummy_input)
        self.assertEqual(out.shape, (self.b, 1))

    def test_pretrained_backbone_is_an_unregistered_interface(self):
        # Extension point for future foundation-model wrappers (e.g. Chronos2): not a
        # runnable model until a subclass loads real weights and registers itself.
        self.assertFalse(is_model_registered("pretrainedbackbonemodel"))
        stub = PretrainedBackboneModel(self.base_cfg)
        with self.assertRaises(NotImplementedError):
            stub(self.dummy_input)


if __name__ == "__main__":
    unittest.main()

