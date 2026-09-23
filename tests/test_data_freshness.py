"""Data freshness tests — download live data and compare to cache.

Run with: pytest tests/test_data_freshness.py --run-network -v
Skipped by default (no network in CI/sandbox).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skfin.dataloaders import DatasetLoader

CACHE_DIR = Path(__file__).resolve().parent.parent / "nbs" / "data"

pytestmark = pytest.mark.network


def _compare_dataframes(cached: pd.DataFrame, fresh: pd.DataFrame, name: str, n_head: int = 5):
    """Compare cached and fresh DataFrames. Assert structure matches, report row delta."""
    assert list(cached.columns) == list(fresh.columns), f"{name}: columns differ"

    # Compare first n_head rows of overlapping range (ignore datetime resolution differences)
    if hasattr(cached.index, 'intersection'):
        overlap_idx = cached.index.intersection(fresh.index)
        if len(overlap_idx) >= n_head:
            head_idx = overlap_idx[:n_head]
            cached_head = cached.loc[head_idx].reset_index(drop=True)
            fresh_head = fresh.loc[head_idx].reset_index(drop=True)
            numeric_cols = cached_head.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                diff = (cached_head[numeric_cols] - fresh_head[numeric_cols]).abs().max().max()
                assert diff < 0.5, (
                    f"{name}: max absolute diff in first {n_head} rows = {diff:.4f} (threshold 0.5)"
                )
                if diff > 0.01:
                    print(f"  {name}: historical values revised (max diff={diff:.4f})")
            else:
                assert cached_head.equals(fresh_head), f"{name}: non-numeric data differs in first {n_head} rows"

    row_delta = len(fresh) - len(cached)
    if row_delta > 0:
        print(f"  {name}: fresh has {row_delta} new rows (expected for live data)")
    elif row_delta < 0:
        print(f"  {name}: WARNING fresh has {-row_delta} FEWER rows")
    else:
        print(f"  {name}: same row count ({len(cached)})")


def _compare_any(cached, fresh, name: str):
    """Compare cached and fresh data — handles DataFrames, dicts, and nested dicts."""
    if isinstance(cached, pd.DataFrame) and isinstance(fresh, pd.DataFrame):
        _compare_dataframes(cached, fresh, name)
    elif isinstance(cached, dict) and isinstance(fresh, dict):
        assert set(cached.keys()) == set(fresh.keys()), f"{name}: keys differ ({set(cached.keys())} vs {set(fresh.keys())})"
        for key in cached:
            _compare_any(cached[key], fresh[key], f"{name}/{key}")
    else:
        pytest.fail(f"{name}: type mismatch (cached={type(cached).__name__}, fresh={type(fresh).__name__})")


# --- Ken French datasets ---

KF_DATASETS = [
    "12_Industry_Portfolios",
    "F-F_Research_Data_Factors",
    "F-F_Momentum_Factor",
    "F-F_Research_Data_Factors_daily",
]


@pytest.mark.parametrize("filename", KF_DATASETS)
def test_kf_freshness(filename, tmp_path):
    """Download Ken French data fresh and compare to cache."""
    cached_loader = DatasetLoader(cache_dir=str(CACHE_DIR))
    fresh_loader = DatasetLoader(cache_dir=str(tmp_path))

    cached = cached_loader.load_kf_returns(filename, force_reload=False)
    try:
        fresh = fresh_loader.load_kf_returns(filename, force_reload=True)
    except Exception as e:
        pytest.skip(f"Download failed: {e}")

    _compare_any(cached, fresh, f"KF:{filename}")


def test_sklearn_stock_returns_freshness(tmp_path):
    """Download sklearn stock returns fresh and compare to cache."""
    cached_loader = DatasetLoader(cache_dir=str(CACHE_DIR))
    fresh_loader = DatasetLoader(cache_dir=str(tmp_path))

    cached = cached_loader.load_sklearn_stock_returns(force_reload=False)
    try:
        fresh = fresh_loader.load_sklearn_stock_returns(force_reload=True)
    except Exception as e:
        pytest.skip(f"Download failed: {e}")

    _compare_dataframes(cached, fresh, "sklearn_stock_returns")


def test_buffett_freshness(tmp_path):
    """Download Buffett 13F data fresh and compare to cache."""
    cached_loader = DatasetLoader(cache_dir=str(CACHE_DIR))
    fresh_loader = DatasetLoader(cache_dir=str(tmp_path))

    cached = cached_loader.load_buffets_data(force_reload=False)
    try:
        fresh = fresh_loader.load_buffets_data(force_reload=True)
    except Exception as e:
        pytest.skip(f"Download failed: {e}")

    _compare_dataframes(cached, fresh, "buffett_13f")


def test_goyal_freshness(tmp_path):
    """Download Goyal characteristics fresh and compare to cache."""
    cached_loader = DatasetLoader(cache_dir=str(CACHE_DIR))
    fresh_loader = DatasetLoader(cache_dir=str(tmp_path))

    cached = cached_loader.load_ag_features(force_reload=False)
    try:
        fresh = fresh_loader.load_ag_features(force_reload=True)
    except Exception as e:
        pytest.skip(f"Download failed: {e}")

    _compare_dataframes(cached, fresh, "goyal_ag")


def test_lm_dictionary_freshness(tmp_path):
    """Download Loughran-McDonald dictionary fresh and compare to cache."""
    cached_loader = DatasetLoader(cache_dir=str(CACHE_DIR))
    fresh_loader = DatasetLoader(cache_dir=str(tmp_path))

    cached = cached_loader.load_loughran_mcdonald_dictionary(force_reload=False)
    try:
        fresh = fresh_loader.load_loughran_mcdonald_dictionary(force_reload=True)
    except Exception as e:
        pytest.skip(f"Download failed: {e}")

    _compare_dataframes(cached, fresh, "lm_dictionary")


