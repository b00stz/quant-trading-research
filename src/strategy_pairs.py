"""
Cointegration-based pairs trading (BTC/ETH), spread modelled as an
Ornstein-Uhlenbeck process:

    dX_t = theta * (mu - X_t) dt + sigma dW_t

Discretised (dt = 1 bar) this is an AR(1):

    X_{t+1} = a + b * X_t + eps_t ,   b = 1 - theta*dt,  a = theta*mu*dt

Fitting an OLS of X_{t+1} on X_t recovers (a, b) and hence:

    theta     = (1 - b) / dt
    mu        = a / (1 - b)
    half_life = ln(2) / theta
    sigma     = std(residuals) / sqrt(dt)          (diffusion coefficient)
    stationary_std = sigma / sqrt(2*theta)

Entry/exit bands are then set at +/- k * stationary_std (or, more robustly
here, at +/- k * rolling std of the spread -- see notes in
`compute_zscore_bands`), which is what the CV bullet's "+/- k standard
deviations" refers to.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OUFit:
    theta: float
    mu: float
    sigma: float
    half_life_bars: float
    stationary_std: float


def engle_granger_hedge_ratio(log_price_a: pd.Series, log_price_b: pd.Series):
    """OLS hedge ratio: log_price_a = alpha + beta*log_price_b + spread."""
    x = log_price_b.values
    y = log_price_a.values
    x1 = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(x1, y, rcond=None)
    alpha, beta = coef
    spread = y - (alpha + beta * x)
    return alpha, beta, pd.Series(spread, index=log_price_a.index, name="spread")


def cointegration_pvalue(log_price_a: pd.Series, log_price_b: pd.Series) -> float:
    from statsmodels.tsa.stattools import coint
    score, pvalue, _ = coint(log_price_a, log_price_b)
    return float(pvalue)


def fit_ou(spread: pd.Series, dt: float = 1.0) -> OUFit:
    s = spread.dropna()
    x_t = s.iloc[:-1].values
    x_t1 = s.iloc[1:].values
    x1 = np.column_stack([np.ones_like(x_t), x_t])
    coef, *_ = np.linalg.lstsq(x1, x_t1, rcond=None)
    a, b = coef
    resid = x_t1 - (a + b * x_t)
    b_clamped = min(b, 0.999999)  # guard against non mean-reverting fits
    theta = (1 - b_clamped) / dt
    mu = a / (1 - b_clamped) if theta > 1e-8 else float(s.mean())
    sigma = float(np.std(resid, ddof=1) / np.sqrt(dt))
    half_life = float(np.log(2) / theta) if theta > 1e-8 else np.inf
    stationary_std = float(sigma / np.sqrt(2 * theta)) if theta > 1e-8 else float(s.std())
    return OUFit(theta=float(theta), mu=float(mu), sigma=sigma,
                 half_life_bars=half_life, stationary_std=stationary_std)


def backtest_pairs(
    price_a: pd.Series,
    price_b: pd.Series,
    lookback: int = 240,      # bars used to (re-)estimate hedge ratio + OU fit
    k_entry: float = 2.0,
    k_exit: float = 0.5,
    round_trip_cost: float = 0.0010,  # 10 bps round trip across both legs
) -> pd.DataFrame:
    log_a = np.log(price_a)
    log_b = np.log(price_b)
    idx = log_a.index

    z = pd.Series(np.nan, index=idx)
    spread_full = pd.Series(np.nan, index=idx)
    ou_half_life = pd.Series(np.nan, index=idx)

    for end in range(lookback, len(idx)):
        window_a = log_a.iloc[end - lookback:end]
        window_b = log_b.iloc[end - lookback:end]
        _, beta, spread_window = engle_granger_hedge_ratio(window_a, window_b)
        ou = fit_ou(spread_window)
        current_spread = log_a.iloc[end] - beta * log_b.iloc[end]
        std_ref = ou.stationary_std if np.isfinite(ou.stationary_std) and ou.stationary_std > 0 else spread_window.std()
        z.iloc[end] = (current_spread - ou.mu) / std_ref if std_ref > 0 else 0.0
        spread_full.iloc[end] = current_spread
        ou_half_life.iloc[end] = ou.half_life_bars

    position = pd.Series(0.0, index=idx)
    pos = 0.0
    for i in range(len(idx)):
        zi = z.iloc[i]
        if np.isnan(zi):
            position.iloc[i] = pos
            continue
        if pos == 0.0:
            if zi >= k_entry:
                pos = -1.0   # spread too high -> short spread (short A, long B)
            elif zi <= -k_entry:
                pos = 1.0    # spread too low -> long spread (long A, short B)
        else:
            if abs(zi) <= k_exit:
                pos = 0.0
        position.iloc[i] = pos

    ret_a = price_a.pct_change().fillna(0.0)
    ret_b = price_b.pct_change().fillna(0.0)
    pos_lag = position.shift(1).fillna(0.0)
    # long spread = long A, short B; short spread = short A, long B
    gross_pnl = pos_lag * (ret_a - ret_b)
    turnover = position.diff().abs().fillna(0.0)
    cost = turnover * round_trip_cost
    net_pnl = gross_pnl - cost

    return pd.DataFrame({
        "z": z,
        "spread": spread_full,
        "ou_half_life_bars": ou_half_life,
        "position": position,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
    })
