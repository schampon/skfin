import numpy as np
import pandas as pd

from skfin.metrics import sharpe_ratio, drawdown


def _monthly_series(mean=0.01, std=0.03, n=60, seed=0):
    np.random.seed(seed)
    return pd.Series(
        np.random.normal(mean, std, n),
        index=pd.date_range("2019-01-31", periods=n, freq="ME"),
    )


def _daily_series(mean=0.0003, std=0.01, n=260, seed=0):
    np.random.seed(seed)
    return pd.Series(
        np.random.normal(mean, std, n),
        index=pd.bdate_range("2022-01-03", periods=n),
    )


def test_sharpe_ratio_monthly_detection():
    pnl = _monthly_series()
    sr = sharpe_ratio(pnl)
    expected = pnl.replace(0, np.nan).mean() / pnl.replace(0, np.nan).std() * np.sqrt(12)
    assert abs(sr - expected) < 1e-10


def test_sharpe_ratio_daily_detection():
    pnl = _daily_series()
    sr = sharpe_ratio(pnl)
    expected = pnl.replace(0, np.nan).mean() / pnl.replace(0, np.nan).std() * np.sqrt(260)
    assert abs(sr - expected) < 1e-10


def test_sharpe_ratio_explicit_override():
    pnl = _monthly_series()
    sr = sharpe_ratio(pnl, num_period_per_year=52)
    expected = pnl.replace(0, np.nan).mean() / pnl.replace(0, np.nan).std() * np.sqrt(52)
    assert abs(sr - expected) < 1e-10


def test_sharpe_ratio_returns_nan_for_unknown_freq():
    pnl = pd.Series(
        [0.01, 0.02, -0.01, 0.005, 0.01],
        index=pd.to_datetime(["2023-01-03", "2023-01-10", "2023-01-25", "2023-02-14", "2023-03-01"]),
    )
    assert np.isnan(sharpe_ratio(pnl))


def test_sharpe_ratio_positive_for_positive_mean():
    pnl = _monthly_series(mean=0.05, std=0.01)
    assert sharpe_ratio(pnl) > 0


def test_drawdown_non_positive():
    pnl = _monthly_series()
    dd = drawdown(pnl, return_in_risk_unit=False)
    assert (dd <= 1e-14).all()


def test_drawdown_zero_at_start():
    pnl = _monthly_series(mean=0.1, std=0.001)
    dd = drawdown(pnl, return_in_risk_unit=False)
    assert dd.iloc[0] <= 0


def test_drawdown_risk_normalized_returns_series():
    pnl = _monthly_series()
    dd = drawdown(pnl, return_in_risk_unit=True, window=12)
    assert isinstance(dd, pd.Series)
    assert len(dd) == len(pnl)
