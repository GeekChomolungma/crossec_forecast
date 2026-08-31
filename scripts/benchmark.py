#!/usr/bin/env python
"""Run the multi-model benchmark defined under `benchmark.models`.

    python scripts/benchmark.py -c experiments/experiment.yaml
    python scripts/benchmark.py -c experiments/experiment.yaml benchmark.top_quantile=0.1 wandb.job_type=benchmark
"""

from _common import base_parser, bootstrap_path, split_known_overrides

bootstrap_path()

from crossec_forecast.pipelines import load_experiment, run_benchmark  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    overrides = split_known_overrides(args.overrides)
    cfg, run_dir, logger = load_experiment(args.config, overrides, job_type="benchmark")
    summary_df = run_benchmark(cfg, run_dir, logger)
    logger.info("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
