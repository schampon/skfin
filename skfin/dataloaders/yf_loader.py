"""Yahoo Finance data loader."""

import logging
from pathlib import Path

import pandas as pd

from skfin.dataloaders.cache import CacheManager

logger = logging.getLogger(__name__)


def load_yf_returns(
    tickers: list,
    start: str = "2010-01-01",
    end: str | None = None,
    force_reload: bool = False,
    cache_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Load price, dividend, and total returns from Yahoo Finance.

    Args:
        tickers: List of stock symbols to load.
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format (default: today).
        force_reload: If True, ignore cache and reload data.
        cache_dir: Cache directory path.

    Returns:
        Dictionary with keys 'price_returns', 'dividend_returns', 'total_returns'.
    """
    cache_manager = CacheManager(cache_dir)
    filename = Path(f"yf_returns_{start}_{end or 'latest'}")

    def loader_func():
        import yfinance as yf

        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            actions=True,
            progress=False,
        )

        data = data.dropna(how="all", axis=1)
        prices = data["Close"]

        if "Dividends" in data.columns.get_level_values(0):
            dividends = data["Dividends"].fillna(0)
            for col in dividends.columns:
                if dividends[col].dtype == object:
                    dividends[col] = dividends[col].apply(
                        lambda x: x.replace(" USD", "") if isinstance(x, str) else x
                    )
                    dividends[col] = dividends[col].astype(float)
            div_returns = dividends.div(prices)
        else:
            div_returns = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        price_returns = prices.pct_change()
        total_returns = price_returns.add(div_returns)

        price_returns = price_returns.dropna(how="all")
        div_returns = div_returns.loc[price_returns.index]
        total_returns = total_returns.loc[price_returns.index]

        return {
            "price_returns": price_returns,
            "dividend_returns": div_returns,
            "total_returns": total_returns,
        }

    return cache_manager.get_cached_dataframe(
        filename=filename,
        loader_func=loader_func,
        force_reload=force_reload,
    )
