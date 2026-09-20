#!/usr/bin/env python3
"""
Run the full three-strategy comparison and print the numbers that go into
the CV bullet. All three strategies are evaluated over the same window:
2020-01-01 to 2023-12-31 (the real funding-rate data's coverage), so the
comparison is apples to apples.

Usage:
    python3 run_backtest.py

Requires (see data/README.md for how to get the two you don't already have):
    data/btc_funding_binance.csv, data/eth_funding_binance.csv   (included)
    data/btc_hourly.csv, data/eth_hourly.csv                     (download)
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.data_io import load_funding, load_ohlcv_cdd
from src.strategy_funding_arb import backtest_funding_arb
from src.strategy_pairs import backtest_pairs, cointegration_pvalue
from src.strategy_orderflow_ml import backtest_orderflow_ml
from src.metrics import summarize

DATA = os.path.join(os.path.dirname(__file__), "data")


def main():
    btc_funding = load_funding(os.path.join(DATA, "btc_funding_binance.csv"))
    eth_funding = load_funding(os.path.join(DATA, "eth_funding_binance.csv"))
    window_start, window_end = btc_funding.index.min(), btc_funding.index.max()

    print("=" * 72)
    print("1) FUNDING-RATE ARBITRAGE  (real Binance funding, 8h settlement)")
    print("=" * 72)
    funding_summaries = {}
    for name, series in [("BTC", btc_funding), ("ETH", eth_funding)]:
        res = backtest_funding_arb(series)
        s = summarize(res["net_pnl"], bars_per_day=3.0, label=f"{name} funding arb")
        funding_summaries[name] = s
        print(s)

    btc_hourly_path = os.path.join(DATA, "btc_hourly.csv")
    eth_hourly_path = os.path.join(DATA, "eth_hourly.csv")
    if not (os.path.exists(btc_hourly_path) and os.path.exists(eth_hourly_path)):
        print("\nHourly OHLCV not found -- see data/README.md to fetch it, "
              "then re-run for strategies 2 and 3.")
        return

    btc = load_ohlcv_cdd(btc_hourly_path)
    eth = load_ohlcv_cdd(eth_hourly_path)
    common = btc.index.intersection(eth.index)
    common = common[(common >= window_start) & (common <= window_end)]
    btc_w, eth_w = btc.loc[common], eth.loc[common]
    print(f"\nJoint price-data window: {common.min()} to {common.max()} ({len(common)} bars)")

    print()
    print("=" * 72)
    print("2) COINTEGRATION-BASED PAIRS TRADING (BTC/ETH)")
    print("=" * 72)
    pval = cointegration_pvalue(np.log(btc_w["close"]), np.log(eth_w["close"]))
    print(f"Full-window Engle-Granger cointegration p-value: {pval:.3f}")
    pairs_res = backtest_pairs(btc_w["close"], eth_w["close"], lookback=240,
                                k_entry=2.0, k_exit=0.5, round_trip_cost=0.0010)
    pairs_summary = summarize(pairs_res["net_pnl"], label="BTC/ETH pairs")
    print(pairs_summary)
    print(f"Median rolling OU half-life: {pairs_res['ou_half_life_bars'].median():.1f} bars")

    print()
    print("=" * 72)
    print("3) ORDER-FLOW-IMBALANCE ML (LightGBM, tick-rule proxy, purged CV)")
    print("=" * 72)
    ml_res = backtest_orderflow_ml(btc_w, horizon=1, n_splits=5, embargo_frac=0.02)
    print(f"OOS purged/embargoed net Sharpe: {ml_res['oos_sharpe_net']:.2f}")
    print(f"Naive (leaky, no purge) in-sample Sharpe: {ml_res['naive_in_sample_sharpe']:.2f}")

    print()
    print("=" * 72)
    print("RANKING (out-of-sample Sharpe, net of costs, 2020-2023, real data)")
    print("=" * 72)
    ranking = [
        ("funding_rate_arb (BTC)", funding_summaries["BTC"]["sharpe"]),
        ("funding_rate_arb (ETH)", funding_summaries["ETH"]["sharpe"]),
        ("cointegration_pairs (BTC/ETH)", pairs_summary["sharpe"]),
        ("orderflow_imbalance_ml (BTC)", ml_res["oos_sharpe_net"]),
    ]
    for name, sharpe in sorted(ranking, key=lambda x: -x[1]):
        verdict = "RETAINED" if sharpe > 1.0 else "REJECTED"
        print(f"  {name:32s}  Sharpe = {sharpe:7.2f}   [{verdict}]")


if __name__ == "__main__":
    main()
