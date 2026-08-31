#!/usr/bin/env python
"""Train + validate + OOS-test a single model.

    python scripts/train.py -c experiments/experiment.yaml
    python scripts/train.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml
    python scripts/train.py -c experiments/experiment.yaml model.name=dlinear train.lr=0.0005 wandb.enabled=false
"""

from _common import base_parser, bootstrap_path, split_known_overrides

bootstrap_path()

from crossec_forecast.pipelines import load_experiment, run_train  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    overrides = split_known_overrides(args.overrides)
    cfg, run_dir, logger = load_experiment(args.config, overrides, job_type="train")
    run_train(cfg, run_dir, logger)


if __name__ == "__main__":
    main()
