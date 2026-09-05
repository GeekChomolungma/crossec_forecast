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
    path: str = ""
    # batch["y"] / model.compute_loss's own target(s) — a list (see BaseClassifierModel
    # .target_dim). NOT the Rank-IC ground truth; that is fwd_ret_col below, kept singular
    # and mandatory on purpose so every model in a run is scored against the same series —
    # never fold a "forward return" into target_cols.
    target_cols: List[str] = field(default_factory=lambda: ["logret1_win"])
    fwd_ret_col: str = "fwd_logret_1"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    feature_pattern: str = r"^crossec_.*_mad_Zscore$"
    feature_cols: Optional[List[str]] = None   # explicit whitelist; overrides feature_pattern
    # Covariate columns (opt-in, default = none). Same precedence as feature_cols/feature_pattern.
    cov_pattern: Optional[str] = None
    cov_cols: Optional[List[str]] = None       # explicit whitelist; overrides cov_pattern
    # Extra-input columns (opt-in): raw/passthrough columns for a model's own internal use
    # (e.g. a raw price series a TSFM wrapper log-diffs itself) — distinct from cov_cols so
    # "feed me this raw column verbatim" doesn't overload feature_cols' established meaning.
    extra_input_pattern: Optional[str] = None
    extra_input_cols: Optional[List[str]] = None  # explicit whitelist; overrides extra_input_pattern
    # `x` packing order: [feature..., cov..., extra_input...] — see PanelTimeSeriesDataset
    # docstring. A model recovers its own slice via self.feature_dim / cov_dim /
    # extra_input_dim (auto-injected alongside seq_len). At least one of
    # feature_cols/cov_cols/extra_input_cols must resolve non-empty.
    seq_len: int = 512
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
    min_delta: float = 1e-3                     # min Rank IC gain over best to reset patience (filters noise)
    # loss selection moved to model.config (model owns model.compute_loss); keys:
    # `loss_type` ("bce" | "focal"), `focal_gamma`, `focal_alpha`.
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
