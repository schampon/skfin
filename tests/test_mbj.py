"""Unit tests for Mbj estimator."""

import numpy as np
from sklearn.base import clone

from skfin.estimators.mbj import Mbj

RNG = np.random.default_rng(42)
N_ASSETS = 5
N_PERIODS = 100
RETURNS = RNG.normal(0.001, 0.02, (N_PERIODS, N_ASSETS))


def test_fit_produces_coef():
    m = Mbj()
    m.fit(RETURNS)
    assert hasattr(m, "coef_")
    assert m.coef_.shape == (N_ASSETS,)


def test_coef_unit_norm_when_rescaled():
    m = Mbj(rescaled_positions=True)
    m.fit(RETURNS)
    assert abs(np.sqrt(np.sum(m.coef_**2)) - 1.0) < 1e-10


def test_coef_not_unit_norm_when_unrescaled():
    m = Mbj(rescaled_positions=False)
    m.fit(RETURNS)
    norm = np.sqrt(np.sum(m.coef_**2))
    assert abs(norm - 1.0) > 1e-6


def test_positive_constraint():
    m = Mbj(positive=True)
    m.fit(RETURNS)
    assert np.all(m.coef_ >= -1e-10)


def test_transform_produces_series():
    m = Mbj()
    m.fit(RETURNS)
    pnl = m.transform(RETURNS)
    assert pnl.shape == (N_PERIODS,)


def test_fit_transform_matches_fit_then_transform():
    m = Mbj()
    pnl1 = m.fit_transform(RETURNS)
    pnl2 = m.fit(RETURNS).transform(RETURNS)
    assert np.allclose(pnl1, pnl2)


def test_predict_returns_coef_row():
    m = Mbj()
    m.fit(RETURNS)
    pred = m.predict(RETURNS)
    assert pred.shape == (1, N_ASSETS)
    assert np.allclose(pred[0], m.coef_)


def test_sklearn_clone():
    m = Mbj(positive=True, rescaled_positions=False)
    m2 = clone(m)
    params = m2.get_params()
    assert params["positive"] is True
    assert params["rescaled_positions"] is False
