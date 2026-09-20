"""
Performance metrics for strategy backtests.
All functions take a pandas Series of PERIOD returns (not prices).
"""
import numpy as np
import pandas as pd


def annualization_factor(bars_per_day: float) -> float:
    return np.sqrt(365.25 * bars_per_day)


def sharpe_ratio(returns: pd.Series, bars_per_day: float = 24.0) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * annualization_factor(bars_per_day))


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    active = r[r != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def cumulative_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((1 + r).prod() - 1.0)


def summarize(returns: pd.Series, bars_per_day: float = 24.0, label: str = "") -> dict:
    return {
        "label": label,
        "n_bars": int(returns.dropna().shape[0]),
        "sharpe": sharpe_ratio(returns, bars_per_day),
        "hit_rate": hit_rate(returns),
        "max_drawdown": max_drawdown(returns),
        "cumulative_return": cumulative_return(returns),
    }
