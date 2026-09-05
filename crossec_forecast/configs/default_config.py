from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Configuration for data loading, filtering, and splitting."""
    # target_cols drives batch["y"] (model.compute_loss's own target(s)) — a list so a
    # model can read more than one column (see BaseClassifierModel.target_dim). Kept
    # STRICTLY SEPARATE from fwd_ret_col below: fwd_ret_col is the one shared, singular,
    # mandatory Rank-IC ground truth every model in a run is scored against — letting
    # models each pick their own "forward return" column would break cross-model Rank IC
    # comparability, so it is never folded into target_cols.
    target_cols: List[str] = field(default_factory=lambda: ["logret1_win"])
    fwd_ret_col: str = "fwd_logret_1"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    feature_pattern: str = r"^crossec_.*_mad_Zscore$"
    feature_cols: Optional[List[str]] = None  # Explicit whitelist; overrides feature_pattern when set
    # Covariate columns — a second, independent column-selection knob, packed into `x`
    # right after the feature columns.
    cov_pattern: Optional[str] = None
    cov_cols: Optional[List[str]] = None      # Explicit whitelist; overrides cov_pattern when set
    # Extra-input columns — a third, independent column-selection knob for RAW/passthrough
    # columns a model needs for its own internal use (e.g. a raw price series a TSFM
    # wrapper log-diffs itself), packed into `x` right after cov. Distinct from cov_cols so
    # a wrapper that wants "give me this raw column verbatim" doesn't have to overload
    # feature_cols (whose established meaning elsewhere is "the crossec_* zscore panel").
    extra_input_pattern: Optional[str] = None
    extra_input_cols: Optional[List[str]] = None  # Explicit whitelist; overrides extra_input_pattern
    # `x` packing convention (see PanelTimeSeriesDataset docstring): [feature..., cov...,
    # extra_input...] concatenated in that fixed order. A model recovers its own slice via
    # self.feature_dim / self.cov_dim / self.extra_input_dim (auto-injected alongside
    # seq_len). At least one of feature_cols/cov_cols/extra_input_cols must resolve
    # non-empty; feature_cols alone is no longer required to be non-empty.
    seq_len: int = 6                     # Lookback window L (e.g. current + 5 past bars)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_steps: int = 1               # Number of timestamp steps to embargo after splits
    batch_size: int = 128
    shuffle_train: bool = True
    num_workers: int = 0
    drop_last: bool = False


@dataclass
class TrainConfig:
    """Configuration for training engine and optimization."""
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    min_delta: float = 1e-3              # min Rank IC gain over best to reset patience (filters noise)
    # NOTE: loss selection now lives in the model config (model.compute_loss reads
    # `loss_type` / `focal_gamma` / `focal_alpha` from its own config dict), so the
    # Trainer is loss-agnostic. See models/base.py.
    device: str = "auto"                 # "auto", "cuda", "mps", "cpu"
    checkpoint_dir: str = "./checkpoints"
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "none"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5


@dataclass
class BenchmarkConfig:
    """Configuration for multi-model benchmark evaluation."""
    models: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "mlp", "config": {"hidden_dims": [64, 32], "dropout": 0.2}},
        {"name": "lstm", "config": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}},
        {"name": "dlinear", "config": {"individual": False}},
    ])
    metrics: List[str] = field(default_factory=lambda: [
        "mean_rank_ic", "ic_ir", "auc", "accuracy", "f1", "top_bottom_spread"
    ])
    top_quantile: float = 0.2
    export_dir: str = "./benchmark_reports"

