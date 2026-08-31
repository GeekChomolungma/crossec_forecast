from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseBacktestEvaluator(ABC):
    """
    Abstract interface for Strategy Robustness and Financial Friction Backtesting.
    Designed for plug-and-play evaluation of slippage, transaction fees,
    turnover penalties, neutralization, Sharpe, Calmar, and Drawdowns.
    """

    @abstractmethod
    def evaluate(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Execute backtest evaluation.
        
        Args:
            predictions_df: DataFrame containing at least:
                ['timestamp', 'symbol', 'pred_prob', 'fwd_logret_1']
                and optionally raw price columns ['open', 'close', etc.]
                
        Returns:
            Dictionary of strategy performance metrics:
            e.g. {'annual_return', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'turnover'}
        """
        pass


class SimpleLongShortBacktester(BaseBacktestEvaluator):
    """
    Standard Quantile Long-Short (Top-K Long, Bottom-K Short) Strategy Backtester
    with customizable trading fee and annualization factor.
    """

    def __init__(
        self,
        top_quantile: float = 0.20,
        fee_rate: float = 0.0005,  # 5 bps per trade
        annualization_factor: float = 365.0,  # default daily
    ):
        self.top_quantile = top_quantile
        self.fee_rate = fee_rate
        self.annualization_factor = annualization_factor

    def evaluate(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        df = predictions_df.copy()
        if "pred_prob" not in df.columns or "fwd_logret_1" not in df.columns:
            raise ValueError("predictions_df must contain 'pred_prob' and 'fwd_logret_1'")

        daily_stats = []

        for ts, group in df.groupby("timestamp", sort=True):
            n = len(group)
            if n < 4:
                continue
            k = max(1, int(n * self.top_quantile))
            sorted_g = group.sort_values(by="pred_prob", ascending=False)

            long_ret = sorted_g.iloc[:k]["fwd_logret_1"].mean()
            short_ret = sorted_g.iloc[-k:]["fwd_logret_1"].mean()

            # Net long-short return after friction
            gross_ls = (long_ret - short_ret) / 2.0
            net_ls = gross_ls - (2.0 * self.fee_rate)  # Simple flat friction per period

            daily_stats.append({
                "timestamp": ts,
                "gross_ls": gross_ls,
                "net_ls": net_ls,
            })

        if not daily_stats:
            return {
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
            }

        res_df = pd.DataFrame(daily_stats)
        rets = res_df["net_ls"].values

        mean_ret = np.mean(rets)
        std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-8

        # Annualized metrics
        ann_return = mean_ret * self.annualization_factor
        ann_vol = std_ret * np.sqrt(self.annualization_factor)
        sharpe = ann_return / (ann_vol + 1e-8) if ann_vol > 0 else 0.0

        # Max Drawdown
        cum_rets = np.cumsum(rets)
        cum_max = np.maximum.accumulate(cum_rets)
        drawdowns = cum_max - cum_rets
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        calmar = ann_return / (max_dd + 1e-8) if max_dd > 0 else 0.0

        return {
            "annual_return": float(ann_return),
            "annual_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
        }

