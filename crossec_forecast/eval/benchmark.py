from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import pandas as pd
import torch

from ..models.registry import build_model, is_model_available
from ..models._optional import ModelDependencyError
from ..configs.default_config import TrainConfig, BenchmarkConfig
from .backtest import SimpleLongShortBacktester
from ..utils.logger import setup_logger
from ..utils.seed import seed_everything


class BenchmarkEngine:
    """
    Automated Multi-Model Benchmark and Horizontal Performance Comparator.
    Executes training, validation, and testing across an arbitrary list of models
    under identical data partitions and outputs comparative reports.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        meta_info: Dict[str, Any],
        models_config: Optional[List[Dict[str, Any]]] = None,
        train_config: Optional[TrainConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None,
        seed: int = 42,
        logger=None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.meta_info = meta_info
        self.train_config = train_config or TrainConfig()
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        self.models_config = models_config or self.benchmark_config.models
        self.seed = seed
        self.logger = logger or setup_logger("BenchmarkEngine")
        self.backtester = SimpleLongShortBacktester(top_quantile=self.benchmark_config.top_quantile)

    def run(self) -> pd.DataFrame:
        """Run benchmark across all configured models."""
        from ..engine.trainer import Trainer

        # OOS test switch. DEFAULT = validation only: the (slower) test predict + evaluate
        # + backtest per model is skipped and the summary carries only val_rank_ic (test /
        # backtest columns NaN), sorted by it. Set CF_BENCH_RUN_TEST=1 to run the full
        # OOS test + backtest and get the complete table.
        run_test = os.getenv("CF_BENCH_RUN_TEST", "").strip().lower() in ("1", "true", "yes", "on")

        self.logger.info("=" * 65)
        self.logger.info("Starting Multi-Model Benchmark Evaluation")
        self.logger.info(f"Feature Dim: {self.meta_info['num_features']}, Sequence Length: {self.meta_info['seq_len']}")
        self.logger.info(f"Samples: Train={self.meta_info['n_train_samples']}, Val={self.meta_info['n_val_samples']}, Test={self.meta_info['n_test_samples']}")
        self.logger.info("=" * 65)

        results = []
        skipped = []

        for m_item in self.models_config:
            model_name = m_item["name"]
            user_m_cfg = m_item.get("config", {})

            # A plugin whose backend library is not importable in this interpreter
            # (e.g. a TSFM whose lib lives in another venv) is skipped, not fatal —
            # run that model from the matching venv. Curate the roster per env, or
            # just let the roster be shared and each env runs the subset it can.
            if not is_model_available(model_name):
                self.logger.warning(
                    f"[skip] '{model_name}' backend not importable in this environment — skipping."
                )
                skipped.append(model_name)
                continue

            # Automatically inject feature_dim / cov_dim / extra_input_dim / target_dim / seq_len
            full_model_cfg = {
                "seq_len": self.meta_info["seq_len"],
                "feature_dim": self.meta_info["num_features"],
                "cov_dim": self.meta_info.get("num_cov", 0),
                "extra_input_dim": self.meta_info.get("num_extra_input", 0),
                "target_dim": self.meta_info.get("num_target", 1),
                "num_classes": 1,
                **user_m_cfg,
            }

            self.logger.info(f"\n>>> Benchmarking Model Plugin: [{model_name.upper()}] <<<")
            if not run_test:
                self.logger.info("OOS test skipped (set CF_BENCH_RUN_TEST=1 to enable) — val metrics only.")
            seed_everything(self.seed)

            # Build model
            try:
                model = build_model(model_name, full_model_cfg)
            except ModelDependencyError as exc:
                self.logger.warning(f"[skip] '{model_name}': {exc}")
                skipped.append(model_name)
                continue

            # Train
            trainer = Trainer(
                model=model,
                config=self.train_config,
                logger=self.logger,
            )
            fit_res = trainer.fit(self.train_loader, self.val_loader)

            # Evaluate on Test (OOS) — only when CF_BENCH_RUN_TEST is set.
            if run_test:
                test_preds_df = trainer.predict(self.test_loader)
                test_metrics = trainer.evaluate(self.test_loader, top_quantile=self.benchmark_config.top_quantile)
                bt_metrics = self.backtester.evaluate(test_preds_df)
            else:
                test_metrics, bt_metrics = {}, {}

            # Classification metrics are only present for output_kind == "binary_prob"
            # models; use .get so point-forecast / other kinds (and val-only runs) still
            # produce a row.
            row = {
                "model": model_name,
                # zero-shot / parameter-free models take the Trainer eval-only path
                # (scored once on val, no training loop) — flag them so the table makes
                # the paradigm difference obvious next to the trained rows.
                "zero_shot": bool(getattr(model, "zero_shot", False)),
                "best_epoch": fit_res["best_epoch"],
                "val_rank_ic": fit_res["best_val_rank_ic"],
                "test_rank_ic": test_metrics.get("mean_rank_ic", float("nan")),
                "test_ic_ir": test_metrics.get("ic_ir", float("nan")),
                "test_auc": test_metrics.get("auc", float("nan")),
                "test_accuracy": test_metrics.get("accuracy", float("nan")),
                "test_f1": test_metrics.get("f1", float("nan")),
                "top_bottom_spread": test_metrics.get("top_bottom_spread", float("nan")),
                "ann_return": bt_metrics.get("annual_return", float("nan")),
                "sharpe_ratio": bt_metrics.get("sharpe_ratio", float("nan")),
                "max_drawdown": bt_metrics.get("max_drawdown", float("nan")),
                "train_time_sec": fit_res["train_time_sec"],
            }
            results.append(row)

        if skipped:
            self.logger.warning(
                f"Skipped {len(skipped)} model(s) with unavailable backends in this env: {skipped}"
            )
        if not results:
            self.logger.warning("No models ran — every roster entry was skipped or failed.")

        res_df = pd.DataFrame(results)
        # Sort by Test Rank IC descending when OOS test ran, else by val Rank IC.
        sort_key = "test_rank_ic" if run_test else "val_rank_ic"
        if not res_df.empty:
            res_df = res_df.sort_values(by=sort_key, ascending=False).reset_index(drop=True)

        # Export reports
        export_path = Path(self.benchmark_config.export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        csv_file = export_path / "benchmark_summary.csv"
        md_file = export_path / "benchmark_summary.md"
        json_file = export_path / "benchmark_summary.json"

        res_df.to_csv(csv_file, index=False)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Model Benchmark Performance Summary\n\n")
            try:
                f.write(res_df.to_markdown(index=False))
            except Exception:
                f.write(res_df.to_string(index=False))
            f.write("\n")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(res_df.to_dict(orient="records"), f, indent=2)

        self.logger.info("\n" + "=" * 65)
        self.logger.info("Benchmark Complete! Summary Table:")
        self.logger.info("\n" + res_df.to_string(index=False))
        self.logger.info(f"Reports saved to: {export_path}")
        self.logger.info("=" * 65)

        return res_df
