from pathlib import Path
import numpy as np
import pandas as pd


def generate_mock_panel_data(
    output_path: str = "mock_standar_panel.csv",
    num_timestamps: int = 120,
    symbols: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic mock financial panel data adhering to the exact standar_panel.csv specification:
    1. Sorted by (timestamp asc, symbol asc).
    2. Contains ~24 crossec_*_mad_Zscore feature columns.
    3. Contains forward lookahead returns fwd_logret_1/3/6 and targets logret1/3/6_win.
    4. Includes asynchronous symbol listing dates (ragged start).
    5. Trailing timestamps have NaNs for future returns & targets.
    """
    np.random.seed(seed)

    if symbols is None:
        symbols = [
            "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
            "DOTUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"
        ]

    # Generate 1-hour timestamps
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [base_time + pd.Timedelta(hours=i) for i in range(num_timestamps)]

    feature_names = [
        "crossec_ema_dev_20_mad_Zscore",
        "crossec_ema_dev_50_mad_Zscore",
        "crossec_macd_mad_Zscore",
        "crossec_macd_signal_mad_Zscore",
        "crossec_macd_hist_mad_Zscore",
        "crossec_adx_14_mad_Zscore",
        "crossec_rsi_14_mad_Zscore",
        "crossec_roc_10_mad_Zscore",
        "crossec_cci_20_mad_Zscore",
        "crossec_natr_14_mad_Zscore",
        "crossec_realized_vol_20_mad_Zscore",
        "crossec_volume_ratio_20_mad_Zscore",
        "crossec_obv_z_20_mad_Zscore",
        "crossec_logret_1_mad_Zscore",
        "crossec_logret_3_mad_Zscore",
        "crossec_logret_6_mad_Zscore",
        "crossec_candle_body_signed_mad_Zscore",
        "crossec_candle_upper_shadow_mad_Zscore",
        "crossec_candle_lower_shadow_mad_Zscore",
        "crossec_tv_st_24_dist_mad_Zscore",
        "crossec_tv_st_24_age_mad_Zscore",
        "crossec_tv_vrb24_pos_mad_Zscore",
        "crossec_tv_vrb24_z_mad_Zscore",
        "crossec_tv_vrb24_width_mad_Zscore",
    ]

    # Simulate listing delays for some symbols (ragged start)
    symbol_start_idx = {
        "BTCUSDT": 0,
        "ETHUSDT": 0,
        "BNBUSDT": 0,
        "SOLUSDT": 10,
        "ADAUSDT": 15,
        "XRPUSDT": 0,
        "DOGEUSDT": 25,
        "AVAXUSDT": 30,
        "DOTUSDT": 0,
        "LINKUSDT": 35,
    }

    rows = []

    for t_idx, ts in enumerate(timestamps):
        active_symbols = [s for s in symbols if t_idx >= symbol_start_idx.get(s, 0)]
        # Sort alphabetically per timestamp slice
        active_symbols.sort()

        # Simulate cross-sectional true signal + noise
        latent_factor = np.random.randn()
        
        slice_records = []
        for s in active_symbols:
            # Latent alpha signal per symbol
            sym_alpha = np.random.randn() * 0.5 + (0.2 if s in ["BTCUSDT", "ETHUSDT"] else -0.1)
            
            # Generate features (correlated with sym_alpha)
            feats = {}
            for f in feature_names:
                feats[f] = np.random.randn() + 0.3 * sym_alpha

            close_val = 100.0 + np.random.uniform(10, 500)
            
            rec = {
                "timestamp": ts,
                "symbol": s,
                "open": close_val - np.random.uniform(0, 2),
                "high": close_val + np.random.uniform(1, 5),
                "low": close_val - np.random.uniform(1, 5),
                "close": close_val,
                "volume": np.random.uniform(100, 10000),
                "close_time": ts + pd.Timedelta(minutes=59, seconds=59),
                "quote_volume": np.random.uniform(10000, 500000),
                "count": np.random.randint(50, 2000),
                "taker_buy_volume": np.random.uniform(50, 5000),
                "taker_buy_quote_volume": np.random.uniform(5000, 250000),
                "_sym_alpha": sym_alpha,
                **feats,
            }
            slice_records.append(rec)

        rows.extend(slice_records)

    df = pd.DataFrame(rows)

    # Sort strictly by (timestamp, symbol)
    df = df.sort_values(by=["timestamp", "symbol"]).reset_index(drop=True)

    # Calculate forward returns and win labels
    df["fwd_logret_1"] = np.nan
    df["fwd_logret_3"] = np.nan
    df["fwd_logret_6"] = np.nan
    df["logret1_win"] = np.nan
    df["logret3_win"] = np.nan
    df["logret6_win"] = np.nan

    # Compute per-symbol forward returns
    for s, group in df.groupby("symbol"):
        idx = group.index
        # Log return = log(close[t+n]) - log(close[t])
        close_series = np.log(group["close"].values)
        for h, fwd_col in [(1, "fwd_logret_1"), (3, "fwd_logret_3"), (6, "fwd_logret_6")]:
            fwd_ret = np.full(len(close_series), np.nan)
            if len(close_series) > h:
                fwd_ret[:-h] = close_series[h:] - close_series[:-h]
            df.loc[idx, fwd_col] = fwd_ret

    # Compute cross-sectional median win labels (0 or 1)
    for ts, group in df.groupby("timestamp"):
        idx = group.index
        for h, fwd_col, win_col in [
            (1, "fwd_logret_1", "logret1_win"),
            (3, "fwd_logret_3", "logret3_win"),
            (6, "fwd_logret_6", "logret6_win"),
        ]:
            rets = group[fwd_col]
            if rets.notna().sum() > 0:
                median_ret = rets.median()
                df.loc[idx, win_col] = (rets > median_ret).astype(float)

    # Drop internal helper column
    df.drop(columns=["_sym_alpha"], inplace=True)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        print(f"Generated mock panel dataset with shape {df.shape} saved to {out_file}")

    return df


if __name__ == "__main__":
    generate_mock_panel_data()

