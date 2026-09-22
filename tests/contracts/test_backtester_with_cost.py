"""Contract tests for BacktesterWithCost — behavioral invariants."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from skfin.backtesting import Backtester
from skfin.backtesting_with_cost import (
    BacktesterWithCost,
    MeanVarianceWithCost,
    compute_batch_holdings_with_cost,
    compute_pnl_components,
)
from skfin.estimators.mean_variance import MeanVariance

RNG = np.random.default_rng(42)
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


def test_degenerates_to_base_without_cost():
    """Without cost, MeanVarianceWithCost matches MeanVariance."""
    V = np.cov(RET.values[:30].T)
    pred = RNG.normal(0.01, 0.02, N_ASSETS)

    from skfin.estimators.mean_variance import compute_batch_holdings

    h_base = compute_batch_holdings(pred, V, A=None, risk_target=None)
    h_cost = compute_batch_holdings_with_cost(pred, V, A=None, vol_liquidity_factor=None)
    assert np.allclose(h_base, h_cost)


def test_cost_reduces_turnover():
    """With vol_liquidity_factor > 0, consecutive holdings are closer together."""
    bt_no_cost = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=None,
        max_train_size=20,
        start_date="2020-02-01",
    )
    bt_no_cost.train(X=RET, y=RET, ret=RET)

    bt_with_cost = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=VLF,
        max_train_size=20,
        start_date="2020-02-01",
    )
    bt_with_cost.train(X=RET, y=RET, ret=RET)

    turnover_no_cost = bt_no_cost.h_.diff().abs().sum(axis=1).mean()
    turnover_with_cost = bt_with_cost.h_.diff().abs().sum(axis=1).mean()
    assert turnover_with_cost < turnover_no_cost


def test_past_holdings_influence():
    """With past_h, solution is pulled toward previous position."""
    V = np.cov(RET.values[:30].T)
    pred = RNG.normal(0.01, 0.02, N_ASSETS)
    vlf = np.full(N_ASSETS, 0.01)
    past_h = np.ones((1, N_ASSETS)) * 5.0

    h_no_past = compute_batch_holdings_with_cost(
        pred, V, A=None, past_h=None, vol_liquidity_factor=vlf
    )
    h_with_past = compute_batch_holdings_with_cost(
        pred, V, A=None, past_h=past_h, vol_liquidity_factor=vlf
    )
    dist_to_past_no = np.linalg.norm(h_no_past - past_h)
    dist_to_past_with = np.linalg.norm(h_with_past - past_h)
    assert dist_to_past_with < dist_to_past_no


def test_pnl_components_keys():
    """When return_pnl_component=True, returns dict with correct keys."""
    bt = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=VLF,
        max_train_size=20,
        start_date="2020-02-01",
        return_pnl_component=True,
    )
    bt.train(X=RET, y=RET, ret=RET)
    assert set(bt.pnl_.keys()) == {"gross", "net = gross - impact cost", "impact cost"}


def test_impact_cost_is_non_positive():
    """Impact costs always reduce returns (cost <= 0)."""
    bt = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=VLF,
        max_train_size=20,
        start_date="2020-02-01",
        return_pnl_component=True,
    )
    bt.train(X=RET, y=RET, ret=RET)
    assert (bt.pnl_["impact cost"] <= 1e-15).all()


def test_net_equals_gross_minus_cost():
    """Net PnL = gross PnL - impact cost."""
    bt = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=VLF,
        max_train_size=20,
        start_date="2020-02-01",
        return_pnl_component=True,
    )
    bt.train(X=RET, y=RET, ret=RET)
    expected_net = bt.pnl_["gross"] + bt.pnl_["impact cost"]
    actual_net = bt.pnl_["net = gross - impact cost"]
    assert np.allclose(actual_net.dropna(), expected_net.dropna())


def test_backtester_with_cost_degenerates_to_backtester():
    """Without cost parameters, BacktesterWithCost matches Backtester PnL."""
    bt_cost = BacktesterWithCost(
        estimator=MeanVarianceWithCost(),
        vol_liquidity_factor=None,
        max_train_size=20,
        start_date="2020-02-01",
    )
    bt_cost.train(X=RET, y=RET, ret=RET)

    bt_base = Backtester(
        estimator=MeanVariance(risk_target=None),
        max_train_size=20,
        start_date="2020-02-01",
    )
    bt_base.train(X=RET, y=RET, ret=RET)

    overlap = bt_cost.pnl_.index.intersection(bt_base.pnl_.index)
    assert np.allclose(bt_cost.pnl_.loc[overlap], bt_base.pnl_.loc[overlap], atol=1e-10)


def test_clone_works():
    """MeanVarianceWithCost can be cloned via sklearn."""
    m = MeanVarianceWithCost()
    m2 = clone(m)
    assert m2.get_params() == m.get_params()
