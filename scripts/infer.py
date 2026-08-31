#!/usr/bin/env python
"""Batch inference with a trained checkpoint over a full panel.

    python scripts/infer.py -c runs/baseline_v1/<run>/config.yaml \
        --checkpoint runs/baseline_v1/<run>/checkpoints/lstmclassifier_best.pt

    # score a fresh panel with the same model/feature spec
    python scripts/infer.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml \
        --checkpoint path/to/lstm_best.pt --data ./data/new_panel.csv
"""

from _common import base_parser, bootstrap_path, split_known_overrides

bootstrap_path()

from crossec_forecast.pipelines import load_experiment, run_infer  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a *_best.pt state_dict")
    parser.add_argument("--data", default=None, help="Override panel path for fresh inference")
    parser.add_argument("--out", default="predictions.csv", help="Output filename inside the run dir")
    args = parser.parse_args()
    overrides = split_known_overrides(args.overrides)

    cfg, run_dir, logger = load_experiment(args.config, overrides, job_type="infer")
    run_infer(
        cfg,
        run_dir,
        checkpoint=args.checkpoint,
        data_path=args.data,
        output_name=args.out,
        logger=logger,
    )


if __name__ == "__main__":
    main()
