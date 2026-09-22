"""Unit and smoke tests for backtesting_with_cost module."""

import numpy as np
import pandas as pd

from skfin.backtesting_with_cost import (
    BacktesterWithCost,
    MeanVarianceWithCost,
    average_holding_period,
    compute_batch_holdings_with_cost,
    compute_pnl_components,
)

RNG = np.random.default_rng(99)
N_ASSETS = 5
N_PERIODS = 80
INDEX = pd.bdate_range("2020-01-01", periods=N_PERIODS)
RET = pd.DataFrame(
    RNG.normal(0.001, 0.02, (N_PERIODS, N_ASSETS)),
    index=INDEX,
    columns=[f"A{i}" for i in range(N_ASSETS)],
)
VLF = pd.DataFrame(
    RNG.uniform(0.001, 0.01, (N_PERIODS, N_ASSETS)),
    index=INDEX,
    columns=[f"A{i}" for i in range(N_ASSETS)],
)


def test_compute_pnl_components_with_cost():
    """Returns dict with three keys when vol_liquidity_factor provided."""
    h = pd.DataFrame(
        RNG.normal(0, 1, (50, N_ASSETS)),
        index=INDEX[:50],
        columns=RET.columns,
    )
    result = compute_pnl_components(h, RET, vol_liquidity_factor=VLF)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"gross", "net = gross - impact cost", "impact cost"}
    assert len(result["gross"]) > 0


def test_compute_pnl_components_without_cost():
    """Returns Series when no vol_liquidity_factor."""
    h = pd.DataFrame(
        RNG.normal(0, 1, (50, N_ASSETS)),
        index=INDEX[:50],
        columns=RET.columns,
    )
    result = compute_pnl_components(h, RET, vol_liquidity_factor=None)
    assert isinstance(result, pd.Series)


def test_compute_batch_holdings_with_cost_no_cost():
    """Without cost parameters, behaves like standard MV."""
    V = np.cov(RET.values[:30].T)
    pred = RNG.normal(0.01, 0.02, N_ASSETS)
    h = compute_batch_holdings_with_cost(pred, V, A=None)
    assert h.shape == (1, N_ASSETS)


def test_compute_batch_holdings_with_cost_with_vlf():
    """With vol_liquidity_factor, returns valid holdings."""
    V = np.cov(RET.values[:30].T)
    pred = RNG.normal(0.01, 0.02, N_ASSETS)
    vlf = np.full(N_ASSETS, 0.005)
    past_h = np.zeros((1, N_ASSETS))
    h = compute_batch_holdings_with_cost(pred, V, A=None, past_h=past_h, vol_liquidity_factor=vlf)
    assert h.shape == (1, N_ASSETS)
    assert np.all(np.isfinite(h))


def test_compute_batch_holdings_with_constraint():
    """Cash-neutral constraint with cost produces finite results."""
    V = np.cov(RET.values[:30].T)
    pred = RNG.normal(0.01, 0.02, N_ASSETS)
    A = np.ones((N_ASSETS, 1))
    vlf = np.full(N_ASSETS, 0.005)
    h = compute_batch_holdings_with_cost(pred, V, A=A, vol_liquidity_factor=vlf)
    assert np.all(np.isfinite(h))


def test_mean_variance_with_cost_fit_predict():
    """MeanVarianceWithCost follows sklearn API."""
    m = MeanVarianceWithCost()
    X = RET.iloc[:30]
    y = RET.iloc[:30]
    m.fit(X, y)
    h = m.predict(RET.iloc[30:31])
    assert h.shape == (1, N_ASSETS)


def test_backtester_with_cost_train():
    """BacktesterWithCost produces h_ and pnl_."""
    bt = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=VLF,
        max_train_size=20,
        start_date="2020-02-01",
    )
    bt.train(X=RET, y=RET, ret=RET)
    assert hasattr(bt, "h_")
    assert hasattr(bt, "pnl_")
    assert isinstance(bt.h_, pd.DataFrame)


def test_backtester_with_cost_no_cost():
    """Without vol_liquidity_factor, pnl_ is a Series."""
    bt = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=None,
        max_train_size=20,
        start_date="2020-02-01",
    )
    bt.train(X=RET, y=RET, ret=RET)
    assert isinstance(bt.pnl_, pd.Series)


def test_average_holding_period_dataframe():
    """Returns finite positive value for multi-asset holdings."""
    h = pd.DataFrame(
        RNG.normal(0, 1, (50, N_ASSETS)),
        index=INDEX[:50],
        columns=RET.columns,
    )
    hp = average_holding_period(h)
    assert np.isfinite(hp)
    assert hp > 0


def test_average_holding_period_series():
    """Works with single-asset Series."""
    h = pd.Series(RNG.normal(0, 1, 50), index=INDEX[:50])
    hp = average_holding_period(h)
    assert np.isfinite(hp)
    assert hp > 0


def test_backtester_with_cost_pnl_components():
    """With return_pnl_component=True, pnl_ is a dict."""
    bt = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=VLF,
        max_train_size=20,
        start_date="2020-02-01",
        return_pnl_component=True,
    )
    bt.train(X=RET, y=RET, ret=RET)
    assert isinstance(bt.pnl_, dict)
    assert "impact cost" in bt.pnl_
