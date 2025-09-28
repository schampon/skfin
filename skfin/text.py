import logging
import sys

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from IPython.display import HTML, display
from skfin.plot import bar

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Pandas and Matplotlib settings
pd.options.display.max_colwidth = None
plt.style.use("seaborn-whitegrid")


def show_text(dataframe, lexica=None, text_column="text", n=2):
    """
    Displays a sample of the text column in a DataFrame with optional lexical highlighting.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the text data.
        lexica (dict): Dictionary containing 'positive' and 'negative' words for highlighting.
        text_column (str): The name of the column containing text data.
        n (int): The number of samples to display. If None, all data is displayed.
    """
    if n is not None:
        dataframe = dataframe.sample(n=n)

    dataframe[text_column] = (
        dataframe[text_column]
        .str.replace("$", "\$", regex=False)
        .str.replace("\n", " ", regex=False)
    )

    if lexica is not None:
        dataframe[text_column] = dataframe[text_column].apply(
            highlight_lexica, lexica=lexica
        )

    display(HTML(dataframe.to_html(escape=False)))


def green_text(text):
    return f"<b><font color='green'>{text}</font></b>"


def red_text(text):
    return f"<b><font color='red'>{text}</font></b>"


def color_text(word, lexica):
    """
    Wraps a word with HTML to color it based on lexica values.

    Args:
        word (str): The word to color.
        lexica (dict): Dictionary containing 'positive' and 'negative' words for highlighting.

    Returns:
        str: HTML-colored word if found in lexica, otherwise the word itself.
    """
    word_lower = word.lower()
    if word_lower in lexica["positive"]:
        return green_text(word)
    elif word_lower in lexica["negative"]:
        return red_text(word)
    return word


def highlight_lexica(text, lexica):
    """
    Highlights lexica in the text by applying HTML coloring.

    Args:
        text (str or list): Text to highlight. If list, it takes the first element.
        lexica (dict): Dictionary containing 'positive' and 'negative' words for highlighting.

    Returns:
        str: Text with colored words according to lexica.
    """
    if isinstance(text, list):
        text = text[0]
    text = text.replace("<br /><br />", "")
    return " ".join(color_text(word, lexica) for word in text.split())


def plot_document_embeddings(embeddings):
    """
    Plots document embeddings using a scatter plot.

    Args:
        embeddings (pd.DataFrame): DataFrame containing embeddings with index as dates.
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

    specific_date = "2020-03-03"
    ax.text(
        x=embeddings.loc[specific_date][0],
        y=embeddings.loc[specific_date][1],
        s=specific_date,
    )


def plot_word_embeddings(embeddings, num_plots=6):
    """
    Plots word embeddings using bar charts.

    Args:
        embeddings (pd.DataFrame): DataFrame containing word embeddings.
        num_plots (int): Number of plots to generate.
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
    coefficients, top_n=40, fontsize=12, rotation=0, title=None, filename=None
):
    """
    Plots positive and negative coefficients of features.

    Args:
        coefficients (pd.Series): Series of coefficients indexed by feature names.
        top_n (int): Number of top positive and negative coefficients to plot.
        fontsize (int): Font size for the y-tick labels.
        rotation (int): Rotation angle for the y-tick labels.
        title (str): Title of the entire figure.
        filename (str): If provided, the plot is saved with this file name.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))

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


def error_analysis_plot(data, lexica, sample_size=5):
    """
    Plots a subset of data with the largest prediction errors, highlighting lexica.

    Args:
        data (pd.DataFrame): DataFrame containing 'label', 'pred', and 'text' columns.
        lexica (dict): Dictionary containing 'positive' and 'negative' words for highlighting.
        sample_size (int): The number of samples to display from the top and bottom errors.
    """
    data = data.assign(diff=lambda df: df["label"] - df["pred"]).sort_values("diff")
    if sample_size is not None:
        data = pd.concat([data.head(sample_size), data.tail(sample_size)])
    show_text(data, lexica)
