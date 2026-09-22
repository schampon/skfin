"""Performance and risk metrics."""

import numpy as np
import pandas as pd


def _test_monthly(df: pd.Series) -> bool:
    """Check if the series has monthly frequency."""
    return int(len(df) / len(df.asfreq("ME"))) == 1


def _test_bday(df: pd.Series) -> bool:
    """Check if the series has business-day frequency."""
    return int(len(df) / len(df.asfreq("B"))) == 1


def _test_day(df: pd.Series) -> bool:
    """Check if the series has calendar-day frequency."""
    return int(len(df) / len(df.asfreq("D"))) == 1


def sharpe_ratio(
    df: pd.Series,
    num_period_per_year: int | None = None,
    remove_zeros: bool = True,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        df: PnL or returns series with a DatetimeIndex.
        num_period_per_year: Annualization factor. Auto-detected from index if None.
        remove_zeros: Replace zeros with NaN before computing (avoids deflating vol).

    Returns:
        Annualized Sharpe ratio, or NaN if frequency cannot be detected.
    """
    if num_period_per_year is None:
        if _test_monthly(df):
            num_period_per_year = 12
        if _test_bday(df):
            num_period_per_year = 260
        if _test_day(df):
            num_period_per_year = 365
        if num_period_per_year is None:
            return np.nan
    if remove_zeros:
        df = df.replace(0, np.nan)
    return df.mean() / df.std() * np.sqrt(num_period_per_year)


def drawdown(
    x: pd.Series,
    return_in_risk_unit: bool = True,
    window: int = 36,
    num_period_per_year: int = 12,
) -> pd.Series:
    """Compute drawdown from peak cumulative PnL.

    Args:
        x: PnL or returns series.
        return_in_risk_unit: Normalize by rolling volatility for cross-strategy comparison.
        window: Rolling window for volatility estimate (in periods).
        num_period_per_year: Annualization factor for the volatility normalization.

    Returns:
        Drawdown series (non-positive values; 0 at peaks).
    """
    dd = x.cumsum().sub(x.cumsum().cummax())
    if return_in_risk_unit:
        return dd.div(x.rolling(window).std().mul(np.sqrt(num_period_per_year)))
    return dd
