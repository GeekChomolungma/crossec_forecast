import unittest
import torch
from crossec_forecast.models import (
    BaseClassifierModel,
    PretrainedBackboneModel,
    register_model,
    build_model,
    list_registered_models,
    is_model_registered,
    is_model_available,
    list_available_models,
    ModelDependencyError,
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

    def test_missing_backend_model_registers_but_does_not_build(self):
        # A wrapper whose backend library lives in another venv must still register
        # (so configs can name it) but must not be buildable here.
        @register_model("needs_absent_lib")
        class NeedsAbsentLib(BaseClassifierModel):
            REQUIRED_MODULES = ("totally_absent_pkg_xyz",)

            def forward(self, x, **kwargs):
                return x.new_zeros((x.size(0), 1))

        self.assertIn("needs_absent_lib", list_registered_models())
        self.assertFalse(is_model_available("needs_absent_lib"))
        self.assertNotIn("needs_absent_lib", list_available_models())
        self.assertIn("mlp", list_available_models())  # pure-python stays available
        with self.assertRaises(ModelDependencyError):
            build_model("needs_absent_lib", self.base_cfg)

    @unittest.skipUnless(
        is_model_available("chronos_bolt_head_only"),
        "chronos backend not installed in this interpreter",
    )
    def test_chronos_bolt_head_only_point_forecast_contract(self):
        cfg = {**self.base_cfg, "model_id": "amazon/chronos-bolt-tiny", "context_length": 8}
        model = build_model("chronos_bolt_head_only", cfg)
        self.assertEqual(model.output_kind, "point_forecast")

        raw = model(self.dummy_input)
        self.assertEqual(raw.shape, (self.b, 1))

        score = model.to_score(raw)
        self.assertEqual(score.shape, (self.b,))

        batch = {"y": torch.randn(self.b, 1)}
        loss = model.compute_loss(raw, batch)
        self.assertEqual(loss.dim(), 0)
        loss.backward()

        # only the trainable head has grads; the frozen backbone is not even a submodule
        head_grads = [p.grad is not None for p in model.head.parameters()]
        self.assertTrue(all(head_grads))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in model.head.parameters()))

    @unittest.skipUnless(
        is_model_available("chronos_bolt_zeroshot"),
        "chronos backend not installed in this interpreter",
    )
    def test_chronos_bolt_zeroshot_point_forecast_contract(self):
        cfg = {**self.base_cfg, "model_id": "amazon/chronos-bolt-tiny", "context_length": 8}
        model = build_model("chronos_bolt_zeroshot", cfg)
        self.assertEqual(model.output_kind, "point_forecast")
        self.assertTrue(model.zero_shot)
        # no trainable parameters -> Trainer runs it eval-only
        self.assertEqual(model.trainable_parameters(), [])

        raw = model(self.dummy_input)
        self.assertEqual(raw.shape, (self.b, 1))
        self.assertTrue(torch.isfinite(raw).all())

        score = model.to_score(raw)
        self.assertEqual(score.shape, (self.b,))
        self.assertTrue(torch.allclose(score, raw.reshape(self.b)))  # sign=+1 default

        loss = model.compute_loss(raw, {"fwd_logret": torch.randn(self.b, 1)})
        self.assertEqual(loss.dim(), 0)

        # state_dict carries only the device-anchor buffer, not Chronos's frozen weights
        self.assertEqual(list(model.state_dict().keys()), ["_dev_anchor"])

    @unittest.skipUnless(
        is_model_available("moment_head_only"),
        "momentfm backend not installed in this interpreter",
    )
    def test_moment_head_only_binary_prob_contract(self):
        cfg = {**self.base_cfg, "model_id": "AutonLab/MOMENT-1-small"}
        model = build_model("moment_head_only", cfg)
        self.assertEqual(model.output_kind, "binary_prob")

        raw = model(self.dummy_input)
        self.assertEqual(raw.shape, (self.b, 1))

        score = model.to_score(raw)
        self.assertEqual(score.shape, (self.b,))
        self.assertTrue(torch.all((score >= 0.0) & (score <= 1.0)))

        batch = {"y": torch.randint(0, 2, (self.b, 1)).float()}
        loss = model.compute_loss(raw, batch)
        self.assertEqual(loss.dim(), 0)
        loss.backward()

        # frozen MOMENT is not a submodule -> only the head is trainable
        self.assertTrue(all(p.grad is not None for p in model.head.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in model.head.parameters()))

    @unittest.skipUnless(
        is_model_available("moment_zeroshot"),
        "momentfm backend not installed in this interpreter",
    )
    def test_moment_zeroshot_anomaly_factor_contract(self):
        model = build_model("moment_zeroshot", {**self.base_cfg, "model_id": "AutonLab/MOMENT-1-small"})
        self.assertEqual(model.output_kind, "anomaly_score")
        self.assertTrue(model.zero_shot)
        # no trainable parameters -> Trainer runs it eval-only
        self.assertEqual(model.trainable_parameters(), [])

        raw = model(self.dummy_input)
        self.assertEqual(raw.shape, (self.b, 1))
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue(torch.all(raw >= 0.0))  # squared / abs reconstruction error

        score = model.to_score(raw)
        self.assertEqual(score.shape, (self.b,))
        self.assertTrue(torch.allclose(score, raw.reshape(self.b)))  # sign=+1 default

        loss = model.compute_loss(raw, {"y": torch.zeros(self.b, 1)})
        self.assertEqual(loss.dim(), 0)

        # state_dict carries only the device-anchor buffer, not MOMENT's frozen weights
        self.assertEqual(list(model.state_dict().keys()), ["_dev_anchor"])

    def test_pretrained_backbone_is_an_unregistered_interface(self):
        # Extension point for future foundation-model wrappers (e.g. Chronos2): not a
        # runnable model until a subclass loads real weights and registers itself.
        self.assertFalse(is_model_registered("pretrainedbackbonemodel"))
        stub = PretrainedBackboneModel(self.base_cfg)
        with self.assertRaises(NotImplementedError):
            stub(self.dummy_input)


if __name__ == "__main__":
    unittest.main()

