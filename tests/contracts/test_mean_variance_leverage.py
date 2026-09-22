"""Contract tests for MeanVarianceWithLeverage — behavioral invariants."""

import numpy as np
from sklearn.base import clone

from skfin.estimators.mean_variance import MeanVariance
from skfin.estimators.mean_variance_leverage import (
    MeanVarianceWithLeverage,
    compute_holdings_with_leverage,
)

N_ASSETS = 5
N_PERIODS = 50
RNG = np.random.default_rng(42)

RET = RNG.normal(0.001, 0.02, (N_PERIODS, N_ASSETS))
PRED = RET.mean(axis=0)
V = np.cov(RET.T)


def test_leverage_bound_is_tight():
    """L1-norm of holdings equals leverage_target (binding constraint)."""
    h = compute_holdings_with_leverage(PRED, V, leverage_target=1.0, risk_target=10.0)
    assert abs(np.abs(h).sum() - 1.0) < 1e-4


def test_sparsity_with_tight_leverage():
    """Tight leverage constraint induces zeros (L1 regularization property)."""
    h = compute_holdings_with_leverage(PRED, V, leverage_target=0.5, risk_target=10.0)
    n_zeros = np.sum(np.abs(h) < 1e-6)
    assert n_zeros >= 1


def test_relaxed_leverage_matches_unconstrained():
    """With very large leverage_target, result approximates unconstrained solution."""
    h_lev = compute_holdings_with_leverage(
        PRED, V, risk_target=1.0, leverage_target=1000.0
    )
    m = MeanVariance(A=None, risk_target=1.0)
    m.fit(X=None, y=RET)
    h_base = m.predict(PRED).squeeze()
    assert np.corrcoef(h_lev, h_base)[0, 1] > 0.95


def test_leverage_reduces_gross_exposure():
    """Leverage constraint reduces gross exposure vs unconstrained."""
    m_base = MeanVariance(A=None, risk_target=1.0)
    m_base.fit(X=None, y=RET)
    h_base = m_base.predict(PRED).squeeze()
    base_leverage = np.abs(h_base).sum()

    target = base_leverage * 0.5
    h_lev = compute_holdings_with_leverage(
        PRED, V, risk_target=1.0, leverage_target=target
    )
    assert np.abs(h_lev).sum() <= target + 1e-5


def test_estimator_inherits_mean_variance():
    """MeanVarianceWithLeverage is a subclass of MeanVariance."""
    assert issubclass(MeanVarianceWithLeverage, MeanVariance)


def test_clone_roundtrip():
    """sklearn clone preserves leverage_target parameter."""
    m = MeanVarianceWithLeverage(leverage_target=2.0, risk_target=0.5)
    m2 = clone(m)
    assert m2.get_params()["leverage_target"] == 2.0
    assert m2.get_params()["risk_target"] == 0.5
