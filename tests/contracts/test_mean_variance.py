"""Contract tests for MeanVariance estimator — behavioral invariants."""

import numpy as np
from sklearn.base import clone

from skfin.estimators.mean_variance import MeanVariance, TimingMeanVariance, compute_batch_holdings

N_ASSETS = 5
N_PERIODS = 50
RNG = np.random.default_rng(42)

# Synthetic data: 50 periods, 5 assets
RET = RNG.normal(0.001, 0.02, (N_PERIODS, N_ASSETS))
PRED = RET.mean(axis=0)
V = np.cov(RET.T)


def test_cash_neutral_constraint():
    """h.sum() == 0 when A='cash-neutral'."""
    m = MeanVariance(A="cash-neutral", risk_target=None)
    m.fit(X=None, y=RET)
    h = m.predict(PRED)
    assert abs(h.sum()) < 1e-10


def test_risk_scaling():
    """sqrt(h.T @ V @ h) == risk_target when risk_target is set."""
    m = MeanVariance(A="cash-neutral", risk_target=1.0)
    m.fit(X=None, y=RET)
    h = m.predict(PRED).squeeze()
    realized_risk = np.sqrt(h @ V @ h)
    assert abs(realized_risk - 1.0) < 1e-8


def test_unconstrained_holdings():
    """h = V^{-1} @ pred when A=None and risk_target=None."""
    m = MeanVariance(A=None, risk_target=None)
    m.fit(X=None, y=RET)
    h = m.predict(PRED).squeeze()
    expected = np.linalg.inv(V) @ PRED
    assert np.allclose(h, expected, atol=1e-10)


def test_predict_shape():
    """predict(X) returns shape (K, N) for K prediction periods."""
    m = MeanVariance(A="cash-neutral", risk_target=1.0)
    m.fit(X=None, y=RET)
    # Batch of 3 predictions
    X_batch = RET[:3]
    h = m.predict(X_batch)
    assert h.shape == (3, N_ASSETS)


def test_fit_stores_covariance():
    """V_ is (N, N) and symmetric after fit()."""
    m = MeanVariance()
    m.fit(X=None, y=RET)
    assert m.V_.shape == (N_ASSETS, N_ASSETS)
    assert np.allclose(m.V_, m.V_.T)


def test_sklearn_clone_roundtrip():
    """clone(estimator).get_params() matches original."""
    m = MeanVariance(A="cash-neutral", risk_target=2.0)
    m2 = clone(m)
    assert m2.get_params()["risk_target"] == 2.0
    assert m2.get_params()["A"] == "cash-neutral"


def test_numpy_array_constraint():
    """Passing A as numpy array works correctly."""
    A = np.ones(N_ASSETS)
    m = MeanVariance(A=A, risk_target=None)
    m.fit(X=None, y=RET)
    h = m.predict(PRED)
    assert abs(h @ A) < 1e-10


def test_timing_mean_variance_basic():
    """TimingMeanVariance scales by inverse variance."""
    y = RNG.normal(0, 0.02, N_PERIODS)
    m = TimingMeanVariance()
    m.fit(X=None, y=y)
    X = np.array([0.01])
    h = m.predict(X)
    expected = X / np.var(y)
    assert np.allclose(h, expected)


def test_timing_mean_variance_clipping():
    """TimingMeanVariance clips positions when a_min/a_max set."""
    y = RNG.normal(0, 0.02, N_PERIODS)
    m = TimingMeanVariance(a_min=-1.0, a_max=1.0)
    m.fit(X=None, y=y)
    X = np.array([10.0])
    h = m.predict(X)
    # Clipped intermediate value, then divided by sqrt(V)
    assert h.item() <= 1.0 / np.sqrt(np.var(y)) + 1e-10
