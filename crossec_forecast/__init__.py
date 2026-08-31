"""
crossec_forecast: High-Performance Cross-Sectional & Time-Series Financial Forecasting Framework.
"""

from .configs import DataConfig, TrainConfig, BenchmarkConfig
from .data import PanelTimeSeriesDataset, build_dataloaders, TimeSplitter
from .models import (
    BaseClassifierModel,
    register_model,
    build_model,
    list_registered_models,
    is_model_registered,
    MLPClassifier,
    LSTMClassifier,
    DLinearClassifier,
    TSFMWrapper,
)
from .engine import Trainer, FocalLoss, get_loss_fn
from .eval import (
    BenchmarkEngine,
    BaseBacktestEvaluator,
    SimpleLongShortBacktester,
    compute_cross_sectional_rank_ic,
    compute_all_metrics,
)
from .utils import setup_logger, seed_everything

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "TrainConfig",
    "BenchmarkConfig",
    "PanelTimeSeriesDataset",
    "build_dataloaders",
    "TimeSplitter",
    "BaseClassifierModel",
    "register_model",
    "build_model",
    "list_registered_models",
    "is_model_registered",
    "MLPClassifier",
    "LSTMClassifier",
    "DLinearClassifier",
    "TSFMWrapper",
    "Trainer",
    "FocalLoss",
    "get_loss_fn",
    "BenchmarkEngine",
    "BaseBacktestEvaluator",
    "SimpleLongShortBacktester",
    "compute_cross_sectional_rank_ic",
    "compute_all_metrics",
    "setup_logger",
    "seed_everything",
]

