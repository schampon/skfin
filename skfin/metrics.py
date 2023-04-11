import numpy as np


def test_monthly(df):
    return int(len(df) / len(df.asfreq("M"))) == 1


def test_bday(df):
    return int(len(df) / len(df.asfreq("B"))) == 1


def test_day(df):
    return int(len(df) / len(df.asfreq("D"))) == 1


def sharpe_ratio(df, num_period_per_year=None):
    num_period_per_year = None
    if test_monthly(df):
        num_period_per_year = 12
    if test_bday(df):
        num_period_per_year = 260
    if test_day(df):
        num_period_per_year = 365
    if num_period_per_year is None:
        return np.nan
    else:
        return df.mean() / df.std() * np.sqrt(num_period_per_year)


def drawdown(x, return_in_risk_unit=True, window=36, num_period_per_year=12):
    dd = x.cumsum().sub(x.cumsum().cummax())
    if return_in_risk_unit:
        return dd.div(x.rolling(window).std().mul(np.sqrt(num_period_per_year)))
    else:
        return dd
