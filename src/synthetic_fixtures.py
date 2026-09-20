"""
SYNTHETIC test fixtures -- NOT real market data.

Used only to smoke-test that the pairs-trading / OU-fitting / LightGBM code
paths run correctly end to end (correct shapes, no leakage bugs, sane
parameter recovery). Any metric computed on this data is a code-correctness
check, never a reported backtest result. Real backtest numbers come from
`data/*_hourly.csv` (real exchange data) once supplied.
"""
import numpy as np
import pandas as pd


def make_synthetic_pair(n_bars: int = 3000, seed: int = 7):
    rng = np.random.default_rng(seed)
    dt = 1.0
    common_trend = np.cumsum(rng.normal(0, 0.004, n_bars))

    # Ornstein-Uhlenbeck spread with KNOWN parameters, so we can check
    # fit_ou() recovers them approximately.
    theta_true, mu_true, sigma_true = 0.03, 0.15, 0.02
    spread = np.zeros(n_bars)
    spread[0] = mu_true
    for t in range(1, n_bars):
        spread[t] = spread[t-1] + theta_true * (mu_true - spread[t-1]) * dt \
            + sigma_true * rng.normal(0, np.sqrt(dt))

    log_price_b = 8.0 + common_trend
    log_price_a = log_price_b + spread  # beta = 1 by construction

    idx = pd.date_range("2026-01-01", periods=n_bars, freq="h")
    price_a = pd.Series(np.exp(log_price_a), index=idx, name="close")
    price_b = pd.Series(np.exp(log_price_b), index=idx, name="close")

    def to_ohlcv(price):
        noise = rng.normal(0, 0.0008, n_bars)
        high = price * (1 + np.abs(noise) + 0.0005)
        low = price * (1 - np.abs(noise) - 0.0005)
        openp = price.shift(1).fillna(price.iloc[0])
        volume = rng.lognormal(mean=6.0, sigma=0.5, size=n_bars)
        trades = rng.lognormal(mean=4.0, sigma=0.4, size=n_bars)
        return pd.DataFrame({
            "open": openp, "high": high, "low": low, "close": price,
            "volume": volume, "quote_volume": volume * price, "trades": trades,
        }, index=idx)

    return to_ohlcv(price_a), to_ohlcv(price_b), dict(theta=theta_true, mu=mu_true, sigma=sigma_true)
