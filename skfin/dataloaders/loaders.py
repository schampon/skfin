import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

from skfin.dataloaders.cache import CacheManager
from skfin.dataloaders.web_utils import WebUtils
from skfin.dataloaders.cleaners import DataCleaner
from skfin.dataloaders.fomc import FomcUtils
from skfin.dataloaders.constants.mappings import symbol_dict
from skfin.dataloaders.io_utils import _download_file_safely

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Class for loading various financial datasets."""

    def __init__(self, cache_dir: str = "data"):
        self.cache_manager = CacheManager(cache_dir)
        self.logger = logging.getLogger(__name__)

    def load_kf_returns(
        self, filename: str = "12_Industry_Portfolios", force_reload: bool = False
    ) -> Dict:
        """
        Load Ken French return data.

        Args:
            filename: Name of the data file to load
            force_reload: If True, ignore cache and reload data

        Returns:
            Dictionary of return data
        """
        if filename == "12_Industry_Portfolios":
            skiprows, multi_df = 11, True
        elif filename == "F-F_Research_Data_Factors":
            skiprows, multi_df = 3, False
        elif filename == "F-F_Momentum_Factor":
            skiprows, multi_df = 13, False
        elif filename == "F-F_Research_Data_Factors_daily":
            skiprows, multi_df = 4, False
        else:
            skiprows, multi_df = 11, True

        def loader_func():
            path = (
                "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                + filename
                + "_CSV.zip"
            )
            files = WebUtils.download_zip_content(path)

            df = pd.read_csv(
                files.open(filename + ".csv"), skiprows=skiprows, index_col=0
            )
            if "daily" in filename:
                return {
                    "Daily": df.iloc[:-1].pipe(
                        lambda x: x.set_index(pd.to_datetime(x.index))
                    )
                }
            else:
                return DataCleaner.clean_kf_dataframes(df, multi_df=multi_df)

        return self.cache_manager.get_cached_dataframe(
            filename=Path(filename), loader_func=loader_func, force_reload=force_reload
        )

    def load_buffets_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load Buffett's portfolio data.

        Args:
            force_reload: If True, ignore cache and reload data

        Returns:
            DataFrame containing Buffett's portfolio data
        """

        def loader_func():
            path = "https://github.com/slihn/buffetts_alpha_R/archive/master.zip"
            files = WebUtils.download_zip_content(path)

            df = pd.read_csv(
                files.open("buffetts_alpha_R-master/ffdata_brk13f.csv"), index_col=0
            )
            df.index = pd.to_datetime(df.index, format="%m/%d/%Y")
            return df

        return self.cache_manager.get_cached_dataframe(
            filename=Path("ffdata_brk13f.parquet"),
            loader_func=loader_func,
            force_reload=force_reload,
        )

    def load_sklearn_stock_returns(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load stock returns data from scikit-learn.

        Args:
            force_reload: If True, ignore cache and reload data

        Returns:
            DataFrame containing stock returns
        """

        def loader_func():
            url = "https://raw.githubusercontent.com/scikit-learn/examples-data/master/financial-data"
            df = (
                pd.concat(
                    {
                        c: pd.read_csv(f"{url}/{c}.csv", index_col=0, parse_dates=True)[
                            "close"
                        ].diff()
                        for c in symbol_dict.keys()
                    },
                    axis=1,
                )
                .asfreq("B")
                .iloc[1:]
            )
            return df

        return self.cache_manager.get_cached_dataframe(
            filename=Path("sklearn_returns.parquet"),
            loader_func=loader_func,
            force_reload=force_reload,
        )

    def load_fomc_statements(
        self,
        add_url: bool = True,
        force_reload: bool = False,
        progress_bar: bool = False,
        from_year: int = 1999,
    ) -> pd.DataFrame:
        """
        Load FOMC statements.

        Args:
            add_url: If True, adds URLs to the output
            force_reload: If True, ignore cache and reload data
            progress_bar: If True, displays a progress bar during loading
            from_year: Year from which to load data

        Returns:
            DataFrame containing FOMC statements
        """

        def loader_func():
            urls = FomcUtils.get_fomc_urls(from_year=from_year)
            if progress_bar:
                urls_ = tqdm(urls)
            else:
                urls_ = urls

            corpus = [
                DataCleaner.bs_cleaner(
                    WebUtils.parse_html(WebUtils.get_response(url).text)
                )
                for url in urls_
            ]

            statements = FomcUtils.feature_extraction(corpus).set_index("release_date")
            if add_url:
                statements = statements.assign(url=urls)
            return statements.sort_index()

        return self.cache_manager.get_cached_dataframe(
            filename=Path("fomc_statements.parquet"),
            loader_func=loader_func,
            force_reload=force_reload,
        )

    def load_loughran_mcdonald_dictionary(
        self, filename: str = None, force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load the Loughran-McDonald dictionary.

        Args:
            filename: Custom filename to use
            force_reload: If True, ignore cache and reload data

        Returns:
            DataFrame containing the dictionary data
        """
        if filename is None:
            filename = "Loughran-McDonald_MasterDictionary_1993-2021.csv"
        filename = Path(filename)

        def loader_func():
            id = "17CmUZM9hGUdGYjCXcjQLyybjTrcjrhik"
            url = f"https://docs.google.com/uc?export=download&confirm=t&id={id}"
            filepath = self.cache_manager.cache_dir / filename

            _download_file_safely(
                url=url,
                filepath=filepath,
                manual_url="https://sraf.nd.edu/loughran-mcdonald-master-dictionary/",
            )

            return pd.read_csv(filepath)

        return self.cache_manager.get_cached_dataframe(
            filename=filename, loader_func=loader_func, force_reload=force_reload
        )

    def load_10X_summaries(
        self, filename: str = None, force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load 10-X summaries.

        Args:
            filename: Custom filename to use
            force_reload: If True, ignore cache and reload data

        Returns:
            DataFrame containing 10-X summaries
        """
        if filename is None:
            filename = "Loughran-McDonald_10X_Summaries_1993-2021.csv"
        filename = Path(filename)

        def loader_func():
            id = "1CUzLRwQSZ4aUTfPB9EkRtZ48gPwbCOHA"
            url = f"https://docs.google.com/uc?export=download&confirm=t&id={id}"
            filepath = self.cache_manager.cache_dir / filename

            _download_file_safely(
                url=url,
                filepath=filepath,
                manual_url="https://sraf.nd.edu/sec-edgar-data/lm_10x_summaries/",
            )

            return pd.read_csv(filepath)

        df = self.cache_manager.get_cached_dataframe(
            filename=filename,
            loader_func=loader_func,
            force_reload=force_reload,
        )
        return df.assign(
            date=lambda x: pd.to_datetime(x.FILING_DATE, format="%Y%m%d")
        ).set_index("date")

    def load_ag_features(
        self,
        filename: str = None,
        sheet_name: str = "Monthly",
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load Amit Goyal's characteristics data.

        Args:
            filename: Custom filename to use
            sheet_name: Name of the sheet to load
            force_reload: If True, ignore cache and reload data

        Returns:
            DataFrame containing characteristic data
        """
        if filename is None:
            filename = "Data2024.xlsx"
        filename = Path(filename)

        def loader_func():
            id = "10_nkOkJPvq4eZgNl-1ys63PzhbnM3S2y"
            url = f"https://docs.google.com/spreadsheets/d/{id}/export?format=xlsx"
            filepath = self.cache_manager.cache_dir / filename

            _download_file_safely(
                url=url,
                filepath=filepath,
                manual_url="https://sites.google.com/view/agoyal145/data-library",
            )

            return pd.read_excel(filepath, sheet_name=sheet_name)

        df = self.cache_manager.get_cached_dataframe(
            filename=filename,
            loader_func=loader_func,
            force_reload=force_reload,
            sheet_name=sheet_name,
        )
        return df.assign(
            date=lambda x: pd.to_datetime(x.yyyymm, format="%Y%m")
        ).set_index("date")
