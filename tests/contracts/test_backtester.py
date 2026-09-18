"""Contract tests for Backtester — behavioral invariants."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from skfin.backtesting import Backtester, compute_pnl, fit_predict
from skfin.estimators.mean_variance import MeanVariance

RNG = np.random.default_rng(42)
N_ASSETS = 5
N_PERIODS = 80

INDEX = pd.bdate_range("2020-01-01", periods=N_PERIODS)
RET = pd.DataFrame(RNG.normal(0.001, 0.02, (N_PERIODS, N_ASSETS)), index=INDEX, columns=[f"A{i}" for i in range(N_ASSETS)])


def test_no_future_leakage():
    """Holdings at time t depend only on data <= t-1."""
    bt = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    bt.compute_holdings(X=RET, y=RET)
    # First holding date should be after at least max_train_size observations
    first_h_date = bt.h_.index[0]
    assert first_h_date >= INDEX[20]


def test_pnl_computation():
    """PnL = h.shift(pred_lag) * ret, summed across assets."""
    bt = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    bt.compute_holdings(X=RET, y=RET)
    bt.compute_pnl(RET)
    expected = compute_pnl(bt.h_, RET, bt.pred_lag)
    overlap = bt.pnl_.index.intersection(expected.index)
    assert np.allclose(bt.pnl_.loc[overlap], expected.loc[overlap], equal_nan=True)


def test_holdings_shape():
    """h_ has same columns as y."""
    bt = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    bt.compute_holdings(X=RET, y=RET)
    assert list(bt.h_.columns) == list(RET.columns)


def test_index_alignment():
    """h_.index and pnl_.index are subsets of X.index."""
    bt = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    bt.compute_holdings(X=RET, y=RET)
    bt.compute_pnl(RET)
    assert bt.h_.index.isin(RET.index).all()
    assert bt.pnl_.index.isin(RET.index).all()


def test_date_filtering():
    """pnl_ is clipped to [start_date, end_date]."""
    bt = Backtester(
        estimator=MeanVariance(), max_train_size=20,
        start_date="2020-03-01", end_date="2020-03-20",
    )
    bt.compute_holdings(X=RET, y=RET)
    bt.compute_pnl(RET)
    assert bt.pnl_.index[0] >= pd.Timestamp("2020-03-01")
    assert bt.pnl_.index[-1] <= pd.Timestamp("2020-03-20")


def test_chainable():
    """compute_holdings() and compute_pnl() return self."""
    bt = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    result = bt.compute_holdings(X=RET, y=RET)
    assert result is bt
    result2 = bt.compute_pnl(RET)
    assert result2 is bt


def test_compute_pnl_formula():
    """compute_pnl = h.shift(lag) * ret, summed across assets."""
    h = pd.DataFrame(RNG.normal(0, 1, (10, 3)), index=INDEX[:10], columns=["a", "b", "c"])
    ret_small = pd.DataFrame(RNG.normal(0, 0.01, (10, 3)), index=INDEX[:10], columns=["a", "b", "c"])
    result = compute_pnl(h, ret_small, pred_lag=1)
    expected = h.shift(1).mul(ret_small).sum(axis=1)
    assert np.allclose(result.dropna(), expected.dropna())


def test_compute_pnl_series():
    """compute_pnl works with single-asset Series."""
    h = pd.Series(RNG.normal(0, 1, 10), index=INDEX[:10])
    ret_single = pd.Series(RNG.normal(0, 0.01, 10), index=INDEX[:10])
    result = compute_pnl(h, ret_single, pred_lag=1)
    expected = h.shift(1).mul(ret_single)
    assert np.allclose(result.dropna(), expected.dropna())


def test_fit_predict_returns_predictions_and_estimator():
    """fit_predict returns (predictions, fitted_estimator)."""
    m = MeanVariance()
    train = np.arange(20)
    test = np.arange(20, 25)
    pred, est = fit_predict(m, RET.values, RET.values, train, test, return_estimator=True)
    assert pred.shape[0] == 5
    assert hasattr(est, "V_")


def test_fit_predict_without_estimator():
    """fit_predict with return_estimator=False returns only predictions."""
    m = MeanVariance()
    train = np.arange(20)
    test = np.arange(20, 25)
    result = fit_predict(m, RET.values, RET.values, train, test, return_estimator=False)
    assert result.shape[0] == 5


def test_train_convenience():
    """train() is equivalent to compute_holdings + compute_pnl."""
    bt1 = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    bt1.compute_holdings(X=RET, y=RET)
    bt1.compute_pnl(RET)

    bt2 = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01")
    bt2.train(X=RET, y=RET, ret=RET)

    assert np.allclose(bt1.pnl_, bt2.pnl_)


def test_name_propagates():
    """name attribute is applied to pnl_ Series."""
    bt = Backtester(estimator=MeanVariance(), max_train_size=20, start_date="2020-02-01", name="test_strat")
    bt.train(X=RET, y=RET, ret=RET)
    assert bt.pnl_.name == "test_strat"
