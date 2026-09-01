from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def compute_cross_sectional_rank_ic(
    preds: Union[np.ndarray, List[float]],
    returns: Union[np.ndarray, List[float]],
    dates: Union[np.ndarray, List[Any]],
    min_symbols_per_cross_section: int = 3,
) -> Tuple[float, float, pd.Series]:
    """
    Compute daily/bar cross-sectional Spearman Rank IC and ICIR.
    
    Args:
        preds: Continuous prediction scores / probabilities [N]
        returns: Realized forward returns (e.g. fwd_logret_1) [N]
        dates: Timestamps corresponding to each prediction [N]
        min_symbols_per_cross_section: Minimum symbols in a time slice to compute Rank IC
        
    Returns:
        mean_rank_ic: Mean Rank IC across all valid timestamps
        ic_ir: Information Ratio of Rank IC (mean / std)
        daily_ic: Pandas Series of Rank IC indexed by date
    """
    df = pd.DataFrame({
        "pred": np.asarray(preds, dtype=float).ravel(),
        "ret": np.asarray(returns, dtype=float).ravel(),
        "date": list(dates),
    })

    daily_ic_dict = {}
    for dt, group in df.groupby("date", sort=True):
        if len(group) < min_symbols_per_cross_section:
            continue
        # If all predictions or all returns are identical in cross-section, correlation is 0
        if group["pred"].nunique() <= 1 or group["ret"].nunique() <= 1:
            daily_ic_dict[dt] = 0.0
            continue
        corr, _ = spearmanr(group["pred"], group["ret"])
        if np.isnan(corr):
            corr = 0.0
        daily_ic_dict[dt] = float(corr)

    if not daily_ic_dict:
        return 0.0, 0.0, pd.Series(dtype=float)

    daily_ic_series = pd.Series(daily_ic_dict)
    mean_ic = float(daily_ic_series.mean())
    std_ic = float(daily_ic_series.std(ddof=1)) if len(daily_ic_series) > 1 else 0.0
    ic_ir = mean_ic / (std_ic + 1e-8) if std_ic > 0 else 0.0

    return mean_ic, ic_ir, daily_ic_series


def compute_top_bottom_spread(
    preds: Union[np.ndarray, List[float]],
    returns: Union[np.ndarray, List[float]],
    dates: Union[np.ndarray, List[Any]],
    top_quantile: float = 0.2,
    min_symbols_per_cross_section: int = 4,
) -> Tuple[float, pd.Series]:
    """
    Compute daily/bar Top-K vs Bottom-K Quantile return spread (Long-Short return).
    """
    df = pd.DataFrame({
        "pred": np.asarray(preds, dtype=float).ravel(),
        "ret": np.asarray(returns, dtype=float).ravel(),
        "date": list(dates),
    })

    daily_spread_dict = {}
    for dt, group in df.groupby("date", sort=True):
        n = len(group)
        if n < min_symbols_per_cross_section:
            continue
        k = max(1, int(n * top_quantile))
        sorted_group = group.sort_values(by="pred", ascending=False)
        top_ret = sorted_group.iloc[:k]["ret"].mean()
        bottom_ret = sorted_group.iloc[-k:]["ret"].mean()
        daily_spread_dict[dt] = float(top_ret - bottom_ret)

    if not daily_spread_dict:
        return 0.0, pd.Series(dtype=float)

    spread_series = pd.Series(daily_spread_dict)
    return float(spread_series.mean()), spread_series


def compute_classification_metrics(
    probs: Union[np.ndarray, List[float]],
    targets: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard classification metrics (AUC, Accuracy, Precision, Recall, F1)."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(targets, dtype=int).ravel()

    # Binary prediction
    binary_preds = (p >= threshold).astype(int)

    acc = float(accuracy_score(y, binary_preds))
    prec = float(precision_score(y, binary_preds, zero_division=0))
    rec = float(recall_score(y, binary_preds, zero_division=0))
    f1 = float(f1_score(y, binary_preds, zero_division=0))

    try:
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


def compute_all_metrics(
    preds: Union[np.ndarray, List[float]],
    targets: Union[np.ndarray, List[float]],
    returns: Union[np.ndarray, List[float]],
    dates: Union[np.ndarray, List[Any]],
    top_quantile: float = 0.2,
    output_kind: str = "binary_prob",
) -> Dict[str, float]:
    """
    Compute the quant metrics suite.

    Rank IC / ICIR / top-bottom spread are always computed — they only need a
    monotone cross-sectional score, so every ``output_kind`` produces them and they
    are what makes heterogeneous models comparable.

    The classification block (AUC / accuracy / precision / recall / F1) assumes
    ``preds`` are calibrated probabilities in [0, 1] against a binary ``targets``
    column, so it is only computed when ``output_kind == "binary_prob"``. For other
    kinds those keys are simply absent (callers must use ``.get`` — see
    ``eval/benchmark.py`` and ``TO_IMPROVE.md`` C1).
    """
    mean_rank_ic, ic_ir, _ = compute_cross_sectional_rank_ic(preds, returns, dates)
    spread, _ = compute_top_bottom_spread(preds, returns, dates, top_quantile=top_quantile)

    metrics = {
        "mean_rank_ic": mean_rank_ic,
        "ic_ir": ic_ir,
        "top_bottom_spread": spread,
    }
    if output_kind == "binary_prob":
        metrics.update(compute_classification_metrics(preds, targets))
    return metrics

