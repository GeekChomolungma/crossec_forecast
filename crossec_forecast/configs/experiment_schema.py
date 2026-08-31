"""
Structured schema for a full experiment (train / validate / test / benchmark / infer).

This is the single source of truth for the `experiment.yaml` layout. It is used as an
OmegaConf structured config so that:
  * every key has a typed default,
  * unknown keys / typos in the YAML raise instead of being silently ignored,
  * `${...}` interpolation and CLI dot-list overrides work out of the box.

The nested dataclasses intentionally mirror the existing `DataConfig` / `TrainConfig` /
`BenchmarkConfig` so the pipeline adapters stay a thin field-for-field copy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentMeta:
    name: str = "default"          # logical experiment name; groups runs & wandb runs
    seed: int = 42


@dataclass
class RunMeta:
    name: Optional[str] = None     # run folder name; auto-generated (<ts>_<model>_<hash>) when null
    output_root: str = "./runs"    # <output_root>/<experiment.name>/<run.name>/
    tags: List[str] = field(default_factory=list)


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_steps: int = 1


@dataclass
class DataSchema:
    path: str = "./data/mock_standar_panel.csv"
    target_col: str = "logret1_win"
    fwd_ret_col: str = "fwd_logret_1"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    feature_pattern: str = r"^crossec_.*_mad_Zscore$"
    feature_cols: Optional[List[str]] = None   # explicit whitelist; overrides feature_pattern
    seq_len: int = 6
    split: SplitConfig = field(default_factory=SplitConfig)
    batch_size: int = 128
    shuffle_train: bool = True
    num_workers: int = 0
    drop_last: bool = False


@dataclass
class TrainSchema:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    loss_type: str = "bce"                     # "bce" | "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    device: str = "auto"                       # "auto" | "cuda" | "mps" | "cpu"
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau" | "cosine" | "none"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5


@dataclass
class ModelSchema:
    name: str = "mlp"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSchema:
    top_quantile: float = 0.2
    models: List[ModelSchema] = field(default_factory=lambda: [
        ModelSchema(name="mlp", config={"hidden_dims": [64, 32], "dropout": 0.2}),
        ModelSchema(name="lstm", config={"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}),
        ModelSchema(name="dlinear", config={"individual": False}),
    ])


@dataclass
class WandbSchema:
    enabled: bool = True
    project: str = "crossec_forecast"
    entity: Optional[str] = None
    mode: str = "online"                       # "online" | "offline" | "disabled"
    group: Optional[str] = None                # defaults to experiment.name when null
    job_type: str = "train"
    name: Optional[str] = None                 # defaults to run folder name when null
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_freq: int = 1                          # log every N epochs (0 disables per-epoch logging)


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    run: RunMeta = field(default_factory=RunMeta)
    data: DataSchema = field(default_factory=DataSchema)
    train: TrainSchema = field(default_factory=TrainSchema)
    model: ModelSchema = field(default_factory=ModelSchema)          # used by train / infer
    benchmark: BenchmarkSchema = field(default_factory=BenchmarkSchema)  # used by benchmark
    wandb: WandbSchema = field(default_factory=WandbSchema)
