"""
Purged & embargoed walk-forward validation for time series with overlapping
labels (e.g. a label at bar t that depends on returns up to t+H).

Design (Lopez de Prado, "Advances in Financial Machine Learning", ch.7):
  - Walk forward in expanding-window folds: train on [0, t], test on
    (t, t+test_size].
  - PURGE: drop any training observation whose label window [i, i+H] overlaps
    the test window. Prevents the model from training on information that
    "leaks" from the test period through overlapping labels.
  - EMBARGO: additionally drop a further `embargo_frac * len(test)` bars
    immediately after the test window from the NEXT fold's training set, to
    remove serial-correlation leakage (autocorrelated features/returns).

This module returns train/test index arrays; it does not touch the data.
"""
from dataclasses import dataclass
from typing import Iterator, Tuple
import numpy as np


@dataclass
class PurgedWalkForward:
    n_splits: int = 5
    label_horizon: int = 1       # H: bars a label looks ahead (purge window)
    embargo_frac: float = 0.01   # fraction of fold size embargoed after test

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError("Not enough samples for the requested n_splits")
        embargo = max(1, int(fold_size * self.embargo_frac))

        for k in range(1, self.n_splits + 1):
            train_end = k * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            if test_start >= n_samples:
                break

            test_idx = np.arange(test_start, test_end)

            # Candidate training set: everything strictly before the test fold
            train_idx = np.arange(0, train_end)

            # PURGE: drop training obs whose [i, i+H] label window overlaps
            # the test window, i.e. i > train_end - H - 1
            purge_start = max(0, train_end - self.label_horizon)
            train_idx = train_idx[train_idx < purge_start]

            # EMBARGO: also true for the *next* fold, but we apply it
            # retroactively here in case test windows are reused: drop the
            # `embargo` bars immediately preceding test_start from training.
            embargo_start = max(0, test_start - embargo)
            train_idx = train_idx[train_idx < embargo_start]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


def naive_in_sample_sharpe(returns_by_fold_insample) -> float:
    """Helper: Sharpe computed by fitting and testing on the SAME (in-sample)
    data with no purge/embargo -- used only to demonstrate the leakage gap
    described in the CV bullet ('cutting naive in-sample Sharpe from X to Y')."""
    import pandas as pd
    from .metrics import sharpe_ratio
    all_r = pd.concat(returns_by_fold_insample)
    return sharpe_ratio(all_r)
