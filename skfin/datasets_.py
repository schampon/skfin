import warnings
import sys
import logging
from typing import Dict, Optional

import pandas as pd

from skfin.dataloaders import DatasetLoader

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kf_returns(
    filename: str = "12_Industry_Portfolios",
    force_reload: bool = False,
    cache_dir: Optional[str] = "data",
) -> Dict:
    """Load Ken French return data."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_kf_returns(filename, force_reload)


def load_buffets_data(
    force_reload: bool = False, cache_dir: Optional[str] = "data"
) -> pd.DataFrame:
    """Load Buffett's portfolio data."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_buffets_data(force_reload)


def load_sklearn_stock_returns(
    force_reload: bool = False, cache_dir: Optional[str] = "data"
) -> pd.DataFrame:
    """Load stock returns data from scikit-learn."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_sklearn_stock_returns(force_reload)


def load_fomc_statements(
    add_url: bool = True,
    force_reload: bool = False,
    progress_bar: bool = False,
    from_year: int = 1999,
    cache_dir: Optional[str] = "data",
) -> pd.DataFrame:
    """Load FOMC statements."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_fomc_statements(add_url, force_reload, progress_bar, from_year)


def load_loughran_mcdonald_dictionary(
    force_reload: bool = False, cache_dir: Optional[str] = "data"
) -> pd.DataFrame:
    """Load the Loughran-McDonald dictionary."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_loughran_mcdonald_dictionary(force_reload)


def load_10X_summaries(
    force_reload: bool = False, cache_dir: Optional[str] = "data"
) -> pd.DataFrame:
    """Load 10-X summaries."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_10X_summaries(force_reload)


def load_ag_features(
    sheet_name: str = "Monthly",
    force_reload: bool = False,
    cache_dir: Optional[str] = "data",
) -> pd.DataFrame:
    """Load Amit Goyal's characteristics data."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_ag_features(sheet_name, force_reload)
