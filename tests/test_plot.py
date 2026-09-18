import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from skfin.plot import set_axis, line, bar, heatmap, scatter


def test_set_axis_creates_figure():
    fig, ax = set_axis()
    assert fig is not None
    assert ax is not None


def test_line_series():
    line(pd.Series(np.ones(10)), legend_sharpe_ratio=False)


def test_line_dataframe():
    line(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}), legend_sharpe_ratio=False)


def test_bar_series():
    bar(pd.Series([1, 2, 3], index=["a", "b", "c"]))


def test_bar_horizontal():
    bar(pd.Series([1, 2, 3], index=["a", "b", "c"]), horizontal=True)


def test_bar_dataframe_grouped():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2]}, index=["a", "b", "c"])
    bar(df, horizontal=True)


def test_bar_dict_of_series():
    bar({"x": pd.Series([1, 2], index=["a", "b"]), "y": pd.Series([3, 1], index=["a", "b"])})


def test_heatmap_basic():
    df = pd.DataFrame(np.random.randn(3, 3), columns=["x", "y", "z"])
    heatmap(df)


def test_scatter_basic():
    scatter(pd.Series([1, 2, 3], index=["a", "b", "c"]))
