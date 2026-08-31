import unittest
import torch
from crossec_forecast.models import (
    BaseClassifierModel,
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
        self.assertIn("tsfm_wrapper", registered)

    def test_all_registered_models_forward_and_predict_proba(self):
        for name in ["mlp", "lstm", "dlinear", "tsfm_wrapper"]:
            model = build_model(name, self.base_cfg)
            logits = model(self.dummy_input)
            self.assertEqual(logits.shape, (self.b, 1), f"Model {name} output shape mismatch")

            probs = model.predict_proba(self.dummy_input)
            self.assertEqual(probs.shape, (self.b, 1), f"Model {name} proba shape mismatch")
            self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))

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


if __name__ == "__main__":
    unittest.main()

