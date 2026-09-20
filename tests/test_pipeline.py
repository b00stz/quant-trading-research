"""
Unit tests. Where a numeric ground truth exists (the OU-process synthetic
fixture), we assert recovery within a tolerance. Where it doesn't, we assert
structural properties (no lookahead leakage, correct shapes, monotonicity).
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.synthetic_fixtures import make_synthetic_pair
from src.strategy_pairs import fit_ou, engle_granger_hedge_ratio, cointegration_pvalue, backtest_pairs
from src.strategy_funding_arb import backtest_funding_arb
from src.validation import PurgedWalkForward
from src.metrics import sharpe_ratio, hit_rate, max_drawdown


@pytest.fixture(scope="module")
def synthetic_pair():
    return make_synthetic_pair(n_bars=3000, seed=7)


def test_ou_fit_recovers_known_parameters(synthetic_pair):
    ohlcv_a, ohlcv_b, true_params = synthetic_pair
    _, beta, spread = engle_granger_hedge_ratio(
        np.log(ohlcv_a["close"]), np.log(ohlcv_b["close"])
    )
    fit = fit_ou(spread)
    true_half_life = np.log(2) / true_params["theta"]
    assert fit.half_life_bars == pytest.approx(true_half_life, rel=0.25)
    assert fit.theta > 0  # mean-reverting, not explosive


def test_cointegrated_series_reject_null(synthetic_pair):
    ohlcv_a, ohlcv_b, _ = synthetic_pair
    pvalue = cointegration_pvalue(np.log(ohlcv_a["close"]), np.log(ohlcv_b["close"]))
    assert pvalue < 0.05  # these two series ARE cointegrated by construction


def test_purged_walk_forward_no_train_test_overlap():
    cv = PurgedWalkForward(n_splits=5, label_horizon=3, embargo_frac=0.02)
    for train_idx, test_idx in cv.split(1000):
        assert set(train_idx).isdisjoint(set(test_idx))
        # purge: no training index should be within `label_horizon` of the
        # test window's start
        assert train_idx.max() < test_idx.min()


def test_purged_walk_forward_folds_are_chronological():
    cv = PurgedWalkForward(n_splits=4, label_horizon=1, embargo_frac=0.01)
    test_starts = [test_idx.min() for _, test_idx in cv.split(2000)]
    assert test_starts == sorted(test_starts)


def test_metrics_on_known_series():
    # a series of all-positive, slightly noisy returns should have Sharpe > 0,
    # hit rate 1.0, and zero drawdown. A perfectly constant series has zero
    # variance, hence an undefined (we return 0.0) Sharpe -- that's the
    # correct behaviour, not what's being tested here.
    r = pd.Series([0.01, 0.012, 0.009, 0.011, 0.010] * 10)
    assert sharpe_ratio(r) > 0
    assert hit_rate(r) == 1.0
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_zero_variance_returns_zero_not_error():
    r = pd.Series([0.01] * 50)
    assert sharpe_ratio(r) == 0.0


def test_funding_arb_position_bounded():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="8h")
    funding = pd.Series(rng.normal(0, 0.0005, 500), index=idx)
    res = backtest_funding_arb(funding)
    assert res["position"].isin([-1.0, 0.0, 1.0]).all()


def test_pairs_backtest_output_shape(synthetic_pair):
    ohlcv_a, ohlcv_b, _ = synthetic_pair
    res = backtest_pairs(ohlcv_a["close"], ohlcv_b["close"], lookback=240)
    assert len(res) == len(ohlcv_a)
    assert res["position"].dropna().isin([-1.0, 0.0, 1.0]).all()
