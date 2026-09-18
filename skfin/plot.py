"""Visualization helpers for quick data exploration."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from skfin.metrics import sharpe_ratio


def set_axis(
    ax: plt.Axes | None = None,
    figsize: tuple = (8, 5),
    title: str | None = None,
    fig: plt.Figure | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create or reuse a matplotlib figure and axes pair.

    Args:
        ax: Existing axes to reuse. If None, creates a new figure.
        figsize: Figure size as (width, height) in inches.
        title: Optional title for the axes.
        fig: Existing figure to associate with the axes.

    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = fig or ax.get_figure()
    if title is not None:
        ax.set_title(title)
    return fig, ax


def line(
    df: pd.DataFrame | pd.Series | dict | list,
    sort: bool = True,
    figsize: tuple = (8, 5),
    ax: plt.Axes | None = None,
    title: str = "",
    cumsum: bool = False,
    loc: str = "center left",
    bbox_to_anchor: tuple | None = (1, 0.5),
    legend_sharpe_ratio: bool | None = None,
    legend: bool = True,
    yscale: str | None = None,
    start_date: str | None = None,
):
    """Plot one or more time series as lines.

    Args:
        df: Data to plot. Accepts DataFrame, Series, dict, or list of Series.
        sort: Sort columns by final value (highest on top).
        figsize: Figure size as (width, height) in inches.
        ax: Existing axes to plot on.
        title: Plot title.
        cumsum: Cumulate the series before plotting.
        loc: Legend location string.
        bbox_to_anchor: Legend anchor point.
        legend_sharpe_ratio: Append Sharpe ratio to legend labels. Auto-enabled with cumsum.
        legend: Whether to show the legend.
        yscale: Y-axis scale (e.g. "log").
        start_date: Trim data before this date.
    """
    df = df.copy() if isinstance(df, (pd.DataFrame, pd.Series)) else df
    if loc == "best":
        bbox_to_anchor = None
    if isinstance(df, (dict, list)):
        df = pd.concat(df, axis=1)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if start_date is not None:
        df = df[start_date:]
    if cumsum and legend_sharpe_ratio is None:
        legend_sharpe_ratio = True
    if legend_sharpe_ratio:
        df.columns = [f"{c}: sr={sharpe_ratio(df[c]): 3.2f}" for c in df.columns]
    if cumsum:
        df = df.cumsum()
    if sort:
        df = df.loc[:, lambda x: x.iloc[-1].sort_values(ascending=False).index]
    if ax is None:
        fig, ax = set_axis(ax=ax, figsize=figsize)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5)
    if title != "":
        ax.set_title(title)
    ax.plot(df.index, df.values)
    if legend:
        ax.legend(df.columns, loc=loc, bbox_to_anchor=bbox_to_anchor)
    if yscale == "log":
        ax.set_yscale("log")


def bar(
    df: pd.DataFrame | pd.Series | dict,
    err: pd.Series | None = None,
    sort: bool = True,
    figsize: tuple = (8, 5),
    ax: plt.Axes | None = None,
    title: str | None = None,
    horizontal: bool = False,
    rotation: int = 0,
):
    """Plot a bar chart from a Series, dict, or multi-column DataFrame.

    Args:
        df: Data to plot. A Series or dict produces single bars. A DataFrame
            with multiple columns produces grouped bars (one group per index label).
        err: Error bars (single-series only, same index as df).
        sort: Sort rows by value (single-series) or by row mean (multi-series).
        figsize: Figure size as (width, height) in inches.
        ax: Existing axes to plot on.
        title: Plot title.
        horizontal: Draw horizontal bars.
        rotation: Tick label rotation angle.
    """
    if isinstance(df, dict):
        df = pd.concat(df, axis=1) if any(isinstance(v, pd.Series) for v in df.values()) else pd.Series(df)
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if sort:
        df = df.loc[df.mean(axis=1).sort_values().index]

    labels = df.index
    x = np.arange(len(labels))
    n_cols = df.shape[1]
    bar_width = 1 / (n_cols + 1)

    fig, ax = set_axis(ax=ax, figsize=figsize, title=title)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i in range(n_cols):
        offset = (i - (n_cols - 1) / 2) * bar_width
        if horizontal:
            ax.barh(
                x + offset, df.iloc[:, i].values,
                height=bar_width, capsize=5,
                xerr=err if n_cols == 1 else None,
            )
        else:
            ax.bar(
                x + offset, df.iloc[:, i].values,
                width=bar_width, capsize=5,
                yerr=err if n_cols == 1 else None,
            )

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(labels, rotation=rotation)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation)

    if n_cols > 1:
        ax.legend(df.columns)
    ax.set_title(title)


def heatmap(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
    figsize: tuple = (8, 5),
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcompute: bool = True,
    cmap: str = "RdBu",
):
    """Plot a color-coded matrix (e.g. correlations, exposures).

    Args:
        df: Matrix to plot. Rows become x-axis, columns become y-axis.
        ax: Existing axes to plot on.
        fig: Existing figure to use.
        figsize: Figure size as (width, height) in inches.
        title: Plot title.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        vcompute: Auto-compute symmetric vmin/vmax from data.
        cmap: Matplotlib colormap name.
    """
    labels_x = df.index
    x = np.arange(len(labels_x))
    labels_y = df.columns
    y = np.arange(len(labels_y))
    if vcompute:
        vmax = df.abs().max().max()
        vmin = -vmax
    fig, ax = set_axis(ax=ax, figsize=figsize, title=title, fig=fig)
    pos = ax.imshow(
        df.T.values, cmap=cmap, interpolation="nearest", vmax=vmax, vmin=vmin
    )
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(labels_x, rotation=90)
    ax.set_yticklabels(labels_y)
    ax.grid(True)
    fig.colorbar(pos, ax=ax)


def scatter(
    df: pd.DataFrame | pd.Series,
    ax: plt.Axes | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xticks: list | None = None,
    yticks: list | None = None,
    figsize: tuple = (8, 5),
    title: str | None = None,
):
    """Plot values against their index as a scatter plot.

    Args:
        df: Series with values on x-axis and index labels on y-axis.
        ax: Existing axes to plot on.
        xscale: X-axis scale (e.g. "log").
        yscale: Y-axis scale (e.g. "log").
        xlabel: X-axis label.
        ylabel: Y-axis label.
        xticks: Custom x-axis tick positions.
        yticks: Custom y-axis tick positions.
        figsize: Figure size as (width, height) in inches.
        title: Plot title.
    """
    fig, ax = set_axis(ax=ax, figsize=figsize, title=title)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.scatter(df, df.index, facecolors="none", edgecolors="b", s=50)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
