"""Smoke tests for StackingBacktester."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skfin.ensemble import StackingBacktester
from skfin.estimators import MeanVariance, Ridge


RNG = np.random.default_rng(42)
N_ASSETS = 5
N_PERIODS = 150
INDEX = pd.date_range("1960-01-01", periods=N_PERIODS, freq="ME")
RET = pd.DataFrame(
    RNG.normal(0.001, 0.02, (N_PERIODS, N_ASSETS)),
    index=INDEX,
    columns=[f"A{i}" for i in range(N_ASSETS)],
)
X = RET.rolling(12).mean().fillna(0)
Y = RET.shift(-1)
START_DATE = "1963-01-01"


def test_stacking_backtester_runs():
    estimators = {
        "momentum": MeanVariance(),
        "ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(), MeanVariance()),
    }
    m = StackingBacktester(
        estimators=estimators, window=30, min_periods=30, start_date=START_DATE
    ).train(X, Y, RET)
    assert hasattr(m, "pnls_")
    assert hasattr(m, "h_")
    assert hasattr(m, "coef_")
    assert "ensemble" in m.pnls_.columns


def test_stacking_backtester_coef_shape():
    estimators = {
        "momentum": MeanVariance(),
        "ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(), MeanVariance()),
    }
    m = StackingBacktester(
        estimators=estimators, window=30, min_periods=30, start_date=START_DATE
    ).train(X, Y, RET)
    assert m.coef_.shape[1] == len(estimators)


def test_stacking_backtester_returns_self():
    estimators = {
        "momentum": MeanVariance(),
        "ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(), MeanVariance()),
    }
    m = StackingBacktester(estimators=estimators, window=30, min_periods=30, start_date=START_DATE)
    result = m.train(X, Y, RET)
    assert result is m
