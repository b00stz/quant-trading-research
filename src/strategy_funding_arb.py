"""
Funding-rate arbitrage (delta-neutral carry).

Idealisation: hold spot + opposite perpetual position sized so price risk is
hedged out; P&L per settlement period is the funding payment itself (funding
is exchanged between longs and shorts on perpetual futures every 8h on
Binance). This ignores basis risk, margin/borrow costs and hedge slippage --
real deployment would need those -- but it isolates the funding-carry edge
that this strategy family actually targets, and every number below comes
from real, exchange-reported funding-rate history (not simulated).

Signal: take a position whose sign matches the trailing mean funding rate
over `lookback` settlement periods, provided the trailing mean clears a
minimum-edge threshold (in per-period terms) large enough to justify paying
the round-trip cost of putting the hedge on. Hold until the signal flips or
decays below threshold, then flip (incurring the round-trip cost again).
"""
import numpy as np
import pandas as pd


def backtest_funding_arb(
    funding: pd.Series,
    lookback: int = 9,          # 9 settlements = 3 days at 8h cadence
    min_edge_annualized: float = 0.03,  # 3% annualized minimum edge to act
    round_trip_cost: float = 0.0006,    # 6 bps round trip (taker fee x2 approx)
    periods_per_year: float = 365.25 * 3,  # 3 settlements/day
) -> pd.DataFrame:
    f = funding.dropna().copy()
    trailing = f.rolling(lookback, min_periods=lookback).mean()
    min_edge_per_period = min_edge_annualized / periods_per_year

    position = pd.Series(0.0, index=f.index)
    active = trailing.abs() >= min_edge_per_period
    position[active] = np.sign(trailing[active])
    position = position.ffill().fillna(0.0)

    # position is decided using info up to t; realize funding paid at t+1
    position_lagged = position.shift(1).fillna(0.0)
    gross_pnl = position_lagged * f

    turnover = position.diff().abs().fillna(0.0)  # 0, 1 or 2 on a flip
    cost = turnover * (round_trip_cost / 2.0)      # cost per unit of turnover

    net_pnl = gross_pnl - cost

    return pd.DataFrame({
        "funding_rate": f,
        "trailing_mean": trailing,
        "position": position,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
    })
