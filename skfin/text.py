"""Text analysis display and visualization helpers."""

import logging

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from IPython.display import HTML, display

from skfin.plot import bar

logger = logging.getLogger(__name__)


def show_text(
    dataframe: pd.DataFrame,
    lexica: dict | None = None,
    text_column: str = "text",
    n: int | None = 2,
):
    """Display a sample of text with optional lexical highlighting.

    Args:
        dataframe: DataFrame containing the text data.
        lexica: Dictionary with 'positive' and 'negative' word lists for highlighting.
        text_column: Name of the column containing text data.
        n: Number of samples to display. None displays all.
    """
    dataframe = dataframe.copy()
    if n is not None:
        dataframe = dataframe.sample(n=n)

    dataframe[text_column] = (
        dataframe[text_column]
        .str.replace("$", r"\$", regex=False)
        .str.replace("\n", " ", regex=False)
    )

    if lexica is not None:
        dataframe[text_column] = dataframe[text_column].apply(
            highlight_lexica, lexica=lexica
        )

    display(HTML(dataframe.to_html(escape=False)))


def green_text(text: str) -> str:
    """Wrap text in bold green HTML tags."""
    return f"<b><font color='green'>{text}</font></b>"


def red_text(text: str) -> str:
    """Wrap text in bold red HTML tags."""
    return f"<b><font color='red'>{text}</font></b>"


def color_text(word: str, lexica: dict) -> str:
    """Color a word green/red if it appears in the lexica, unchanged otherwise.

    Args:
        word: Word to potentially color.
        lexica: Dict with 'positive' and 'negative' word sets.
    """
    word_lower = word.lower()
    if word_lower in lexica["positive"]:
        return green_text(word)
    elif word_lower in lexica["negative"]:
        return red_text(word)
    return word


def highlight_lexica(text: str | list, lexica: dict) -> str:
    """Apply sentiment coloring to every word in a text string.

    Args:
        text: Input text (or single-element list). HTML line breaks are stripped.
        lexica: Dict with 'positive' and 'negative' word sets.

    Returns:
        HTML string with colored words.
    """
    if isinstance(text, list):
        text = text[0]
    text = text.replace("<br /><br />", "")
    return " ".join(color_text(word, lexica) for word in text.split())


def plot_document_embeddings(
    embeddings: pd.DataFrame,
    highlight_date: str | None = None,
):
    """Plot document embeddings as a scatter plot colored by year.

    Args:
        embeddings: DataFrame with 2 columns (PC0, PC1) and a DatetimeIndex.
        highlight_date: Optional date string to annotate on the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_years = [str(year) for year in embeddings.index.year.unique()]
    colors = cm.RdBu(np.linspace(0, 1, len(unique_years)))

    for i, year in enumerate(unique_years):
        ax.scatter(
            x=embeddings.loc[year][0], y=embeddings.loc[year][1], color=colors[i]
        )

    ax.legend(unique_years, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")

    if highlight_date is not None and highlight_date in embeddings.index:
        ax.text(
            x=embeddings.loc[highlight_date][0],
            y=embeddings.loc[highlight_date][1],
            s=highlight_date,
        )


def plot_word_embeddings(embeddings: pd.DataFrame, num_plots: int = 6):
    """Plot top words per topic dimension as horizontal bar charts.

    Args:
        embeddings: DataFrame where each column is a topic dimension.
        num_plots: Number of topic dimensions to plot.
    """
    fig, axes = plt.subplots(
        nrows=num_plots // 2, ncols=2, figsize=(20, 16), sharex=True
    )
    plt.subplots_adjust(wspace=0.5)
    axes = axes.ravel()

    for i in range(num_plots):
        top_words = embeddings[i].sort_values(ascending=False).head(10)
        bar(top_words, horizontal=True, ax=axes[i], title=f"Topic {i}")


def coefs_plot(
    coefficients: pd.Series | pd.DataFrame,
    top_n: int = 40,
    fontsize: int = 12,
    rotation: int = 0,
    title: str | None = None,
    filename: str | None = None,
):
    """Plot positive and negative feature coefficients side by side.

    Args:
        coefficients: Model coefficients indexed by feature name.
        top_n: Number of top coefficients to show per side.
        fontsize: Font size for feature labels.
        rotation: Label rotation angle.
        title: Figure suptitle.
        filename: If provided, save the plot as {filename}.png.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.5)

    coefficients = coefficients.squeeze()
    positive_coefs = (
        coefficients.loc[coefficients > 0].sort_values(ascending=True).head(top_n)
    )
    negative_coefs = (
        coefficients.loc[coefficients < 0].sort_values(ascending=False).tail(top_n)
    )

    axes[0].barh(np.arange(len(negative_coefs)), negative_coefs.values, capsize=5)
    axes[0].set_yticks(np.arange(len(negative_coefs)))
    axes[0].set_yticklabels(negative_coefs.index, rotation=rotation, fontsize=fontsize)
    axes[0].set_title("Negative Coefficients")

    axes[1].barh(np.arange(len(positive_coefs)), positive_coefs.values, capsize=5)
    axes[1].set_yticks(np.arange(len(positive_coefs)))
    axes[1].set_yticklabels(positive_coefs.index, rotation=rotation, fontsize=fontsize)
    axes[1].yaxis.tick_right()
    axes[1].set_title("Positive Coefficients")

    if title:
        fig.suptitle(title, y=0.92)

    if filename:
        plt.savefig(f"{filename}.png", orientation="landscape", bbox_inches="tight")


def error_analysis_plot(
    data: pd.DataFrame,
    lexica: dict,
    sample_size: int | None = 5,
):
    """Display samples with the largest prediction errors, highlighting lexica.

    Args:
        data: DataFrame with 'label', 'pred', and 'text' columns.
        lexica: Dict with 'positive' and 'negative' word sets for highlighting.
        sample_size: Number of worst predictions to show from each tail.
    """
    data = data.assign(diff=lambda df: df["label"] - df["pred"]).sort_values("diff")
    if sample_size is not None:
        data = pd.concat([data.head(sample_size), data.tail(sample_size)])
    show_text(data, lexica)
