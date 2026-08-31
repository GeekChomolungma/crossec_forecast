from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Configuration for data loading, filtering, and splitting."""
    target_col: str = "logret1_win"
    fwd_ret_col: str = "fwd_logret_1"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    feature_pattern: str = r"^crossec_.*_mad_Zscore$"
    feature_cols: Optional[List[str]] = None  # Explicit whitelist; overrides feature_pattern when set
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
    loss_type: str = "bce"               # "bce" or "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
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

