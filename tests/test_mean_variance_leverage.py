"""Unit tests for mean_variance_leverage module."""

import numpy as np
from sklearn.base import clone

from skfin.estimators.mean_variance_leverage import (
    MeanVarianceWithLeverage,
    compute_holdings_with_leverage,
)

RNG = np.random.default_rng(99)
N_ASSETS = 5
RET = RNG.normal(0.001, 0.02, (50, N_ASSETS))
V = np.cov(RET.T)
PRED = RET.mean(axis=0)


def test_compute_holdings_with_leverage_returns_array():
    h = compute_holdings_with_leverage(PRED, V, leverage_target=1.0)
    assert isinstance(h, np.ndarray)
    assert h.shape == (N_ASSETS,)


def test_leverage_constraint_respected():
    h = compute_holdings_with_leverage(PRED, V, leverage_target=1.0)
    assert np.abs(h).sum() <= 1.0 + 1e-6


def test_risk_constraint_respected():
    h = compute_holdings_with_leverage(PRED, V, risk_target=0.5)
    realized_var = h @ V @ h
    assert realized_var <= 0.5 + 1e-6


def test_neutrality_constraint():
    A = np.ones((N_ASSETS, 1))
    h = compute_holdings_with_leverage(PRED, V, A=A, leverage_target=2.0)
    assert abs(h.sum()) < 1e-4


def test_estimator_fit_predict():
    m = MeanVarianceWithLeverage(leverage_target=1.0)
    m.fit(X=None, y=RET)
    h = m.predict(PRED)
    assert h.shape == (1, N_ASSETS)


def test_estimator_sklearn_clone():
    m = MeanVarianceWithLeverage(leverage_target=2.5, risk_target=1.0)
    m2 = clone(m)
    params = m2.get_params()
    assert params["leverage_target"] == 2.5
    assert params["risk_target"] == 1.0
