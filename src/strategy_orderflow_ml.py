"""
Order-book-imbalance signal, ML (LightGBM) version.

DATA CAVEAT (important, read before trusting any number this module
produces): genuine order-book-imbalance needs L1/L2 book snapshots or at
minimum a taker buy/sell volume split. Free hourly OHLCV does not carry
either, so this module builds a PROXY order-flow-imbalance feature using the
Lee-Ready "tick rule": each bar's volume is signed by the direction of that
bar's price change, then aggregated. This is a standard textbook
approximation but is a proxy, not real book imbalance -- treat any Sharpe
figure produced by this module as illustrative of the pipeline's mechanics,
not as a validated production edge, until it is re-run on genuine taker
buy/sell volume or L2 data (Binance kline files DO carry a real taker-buy-
base-asset-volume column; if you download the raw klines, e.g. via the
Binance data vision archive, this module can be pointed at the real field
by replacing `signed_volume_proxy` with `2*taker_buy_volume - volume`).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .validation import PurgedWalkForward
from .metrics import sharpe_ratio


def build_features(ohlcv: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = ohlcv.copy()
    ret = df["close"].pct_change()
    tick_sign = np.sign(df["close"].diff()).replace(0, np.nan).ffill().fillna(0.0)
    signed_volume_proxy = tick_sign * df["volume"]

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = ret
    feat["ret_3"] = df["close"].pct_change(3)
    feat["ret_6"] = df["close"].pct_change(6)
    feat["vol_z"] = (df["volume"] - df["volume"].rolling(48).mean()) / df["volume"].rolling(48).std()
    feat["ofi_proxy"] = signed_volume_proxy.rolling(3).sum() / df["volume"].rolling(3).sum().replace(0, np.nan)
    feat["ofi_proxy_slow"] = signed_volume_proxy.rolling(12).sum() / df["volume"].rolling(12).sum().replace(0, np.nan)
    feat["realized_vol_12"] = ret.rolling(12).std()
    feat["hl_range"] = (df["high"] - df["low"]) / df["close"]
    feat["trades_z"] = (df["trades"] - df["trades"].rolling(48).mean()) / df["trades"].rolling(48).std()

    # label: forward return over `horizon` bars (this is what needs purging)
    feat["fwd_ret"] = df["close"].pct_change(horizon).shift(-horizon)
    return feat


def backtest_orderflow_ml(
    ohlcv: pd.DataFrame,
    horizon: int = 1,
    n_splits: int = 5,
    embargo_frac: float = 0.02,
    signal_threshold_quantile: float = 0.6,
    round_trip_cost: float = 0.0008,
) -> dict:
    feat = build_features(ohlcv, horizon=horizon)
    feature_cols = [c for c in feat.columns if c != "fwd_ret"]
    data = feat.dropna(subset=feature_cols + ["fwd_ret"])
    X = data[feature_cols].values
    y = data["fwd_ret"].values
    idx = data.index

    cv = PurgedWalkForward(n_splits=n_splits, label_horizon=horizon, embargo_frac=embargo_frac)
    oos_pred = pd.Series(np.nan, index=idx)
    naive_in_sample_preds = []  # for the "naive vs purged Sharpe" comparison

    for train_idx, test_idx in cv.split(len(idx)):
        model = LGBMRegressor(
            n_estimators=200, max_depth=4, num_leaves=15,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbosity=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        oos_pred.iloc[test_idx] = preds

        # naive (leaky) comparison: fit and "test" on the same in-sample slice
        naive_model = LGBMRegressor(
            n_estimators=200, max_depth=4, num_leaves=15,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbosity=-1,
        )
        # use ALL data up to test_end, including the bars that should be purged
        leaky_train_idx = np.arange(0, test_idx[-1] + 1)
        naive_model.fit(X[leaky_train_idx], y[leaky_train_idx])
        naive_preds = naive_model.predict(X[test_idx])
        naive_in_sample_preds.append(pd.Series(naive_preds, index=idx[test_idx]))

    valid = oos_pred.dropna().index
    signal = oos_pred.loc[valid]
    fwd_ret = data.loc[valid, "fwd_ret"]

    long_th = signal.quantile(signal_threshold_quantile)
    short_th = signal.quantile(1 - signal_threshold_quantile)
    position = pd.Series(0.0, index=valid)
    position[signal >= long_th] = 1.0
    position[signal <= short_th] = -1.0

    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * round_trip_cost
    gross_pnl = position * fwd_ret
    net_pnl = gross_pnl - cost

    naive_all = pd.concat(naive_in_sample_preds)
    naive_signal_long_th = naive_all.quantile(signal_threshold_quantile)
    naive_signal_short_th = naive_all.quantile(1 - signal_threshold_quantile)
    naive_position = pd.Series(0.0, index=naive_all.index)
    naive_position[naive_all >= naive_signal_long_th] = 1.0
    naive_position[naive_all <= naive_signal_short_th] = -1.0
    naive_fwd_ret = data.loc[naive_all.index, "fwd_ret"]
    naive_pnl = naive_position * naive_fwd_ret  # no cost, gross, to isolate the leakage effect

    return {
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "position": position,
        "oos_sharpe_net": sharpe_ratio(net_pnl),
        "naive_in_sample_sharpe": sharpe_ratio(naive_pnl),
        "n_test_bars": int(len(valid)),
    }
