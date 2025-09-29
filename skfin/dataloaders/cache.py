import logging
import sys
from pathlib import Path
from typing import Any, Union, Callable

import pandas as pd

from skfin.dataloaders.io_utils import clean_directory_path, load_dict, save_dict


logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages loading and saving data to cache."""

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = clean_directory_path(cache_dir)
        self.logger = logging.getLogger(__name__)

    def get_cached_dataframe(
        self,
        filename: Union[str, Path],
        loader_func: Callable,
        force_reload: bool = False,
        **kwargs,
    ) -> Any:
        """
        Load a DataFrame from cache or get it via a loader function.

        Args:
            filename: Cache filename
            loader_func: Function to call to load data if not present in cache
            force_reload: If True, ignore cache and reload data
            **kwargs: Additional arguments to pass to loader_func

        Returns:
            Loaded DataFrame or other object
        """

        if isinstance(filename, str):
            filename = Path(filename)

        full_path = self.cache_dir / filename

        if (full_path.exists()) & (not force_reload):
            self.logger.info(f"Loading from cache : {full_path}")
            if filename.suffix == ".parquet":
                return pd.read_parquet(full_path)
            elif filename.suffix == ".csv":
                return pd.read_csv(full_path)
            elif filename.suffix == ".xlsx":
                if "sheet_name" in kwargs:
                    return pd.read_excel(full_path, sheet_name=kwargs["sheet_name"])
                else:
                    return pd.read_excel(full_path)
            else:
                return load_dict(full_path)
        else:
            self.logger.info("Loading from an external source")
            data = loader_func(**kwargs)
            self.save_to_cache(data, full_path)
            return data

    def save_to_cache(self, data: Any, path: Path) -> None:
        """Saving to cache."""
        self.logger.info(f"Saving to the cache : {path}")
        if isinstance(data, pd.DataFrame):
            if path.suffix == ".parquet":
                data.to_parquet(path)
            elif path.suffix == ".csv":
                data.to_csv(path)
        else:
            save_dict(data, path)
