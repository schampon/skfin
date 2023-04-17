import re
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("seaborn-whitegrid")

from IPython.display import HTML, display

pd.options.display.max_colwidth = None

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def show_text(df, lexica=None, text_column="text", n=2):
    if n is not None:
        df = df.sample(n=n)
    df = df.assign(
        **{
            text_column: lambda x: x[text_column]
            .str.replace("$", "\$", regex=False)
            .str.replace("\n", " ", regex=False)
        }
    )
    if lexica is not None:
        df = df.assign(
            **{
                text_column: lambda x: x[text_column].apply(
                    highlight_lexica, lexica=lexica
                )
            }
        )
    display(HTML(df.to_html(escape=False)))


green_text = lambda x: f"<b><font color = green>{x}</font></b>"
red_text = lambda x: f"<b><font color = red>{x}</font></b>"


def color_text(x, lexica):
    if x.lower() in lexica["positive"]:
        return green_text(x)
    elif x.lower() in lexica["negative"]:
        return red_text(x)
    else:
        return x


def highlight_lexica(string, lexica):
    if isinstance(string, list):
        string = string[0]
    string = string.replace("<br /><br />", "")
    return " ".join([color_text(x, lexica) for x in string.split(" ")])


def coefs_plot(coef, n=40, fontsize=12, rotation=0, title=None, filename=None):
    """
    plot the coefficients from a  tfidf+linear_model pipeline on words (with positive and negative values)
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    df_pos = coef.squeeze().loc[lambda x: x > 0].sort_values().tail(n)
    labels = df_pos.index
    x = np.arange(len(labels))
    ax[1].barh(x, df_pos.values, capsize=5)
    ax[1].set_yticks(x)
    ax[1].set_yticklabels(labels, rotation=rotation, fontsize=fontsize)
    ax[1].yaxis.tick_right()
    ax[1].set_title("positive coefficients")

    df_neg = coef.squeeze().loc[lambda x: x < 0].sort_values(ascending=False).tail(n)
    labels = df_neg.index
    x = np.arange(len(labels))
    ax[0].barh(x, df_neg.values, capsize=5)
    ax[0].set_yticks(x)
    ax[0].set_yticklabels(labels, rotation=rotation, fontsize=fontsize)
    ax[0].set_title("negative coefficients")
    if title is not None:
        fig.suptitle(title, y=0.92)
    if filename is not None:
        plt.savefig(
            str(filename) + ".png", orientation="landscape", bbox_inches="tight"
        )


def error_analysis_plot(data, lexica, n=5):
    error_analysis = (
        data.assign(diff=lambda x: x["label"] - x["pred"])
        .sort_values("diff")
        .loc[:, ["label", "pred", "text"]]
    )
    if n is not None:
        error_analysis = error_analysis.pipe(
            lambda x: pd.concat([x.head(n), x.tail(n)])
        )
    show_text(error_analysis, lexica)
