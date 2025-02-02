import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tte.utils.misc import load_vars

ENDPOINTS = load_vars().ENDPOINTS


def load_results():
    df = pd.DataFrame(
        [
            (
                pickle.load(
                    open(f"results/demo/{endpoint}.pkl", "rb"),
                )["global_c_index"],
                pickle.load(open(f"results/baseline_downstream/{endpoint}.pkl", "rb"))[
                    "global_c_index"
                ],
                pickle.load(open(f"results/attention_downstream/{endpoint}.pkl", "rb"))[
                    "global_c_index"
                ],
            )
            for endpoint in ENDPOINTS
        ],
        columns=["demo", "baseline", "attention"],
        index=ENDPOINTS,
    )
    return df.loc[(df.attention - df.baseline).sort_values().index]


# MARKER_DICT = {'attention': '.',  # 'o' is the marker for dots
#                'baseline': 's',  # 's' is the marker for squares
#                'demo': 'o'}  # '^' is the marker for triangle
def both_c_index(df, save_fn, add_mean_line=True, col1='attention', col2='baseline'):
    df = df.copy()
    means = df.mean()
    style()
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, sharey=True)
    if add_mean_line:
        df.loc["------"] = 0.0
        df.loc["mean"] = means  # df.head(-1).mean()
    df1 = df.copy()
    df1.columns = [f"{col} ({means[col]:.3f})" for col in df.columns]

    df1 = df1.melt(
        var_name="Model",
        value_name="C-Index",
        ignore_index=False,
    )

    sns.scatterplot(
        df1,
        x="C-Index",
        y=df1.index,
        hue="Model",
        s=100,
        # style="Model",
        # markers=MARKER_DICT,
        ax=ax[0],
    )
    ax[0].set_xlim(0.5, 1)

    diff = pd.DataFrame(
        df[col1] - df[col2],
        columns=[f"Difference in C-Index ({col1} - {col2})"],
        index=df.index,
    )
    diff.loc["------"] = 9999.0
    sns.scatterplot(
        diff,
        x=diff.columns[0],
        y=diff.index,
        # hue="Model",
        s=100,
        # style="Model",
        # markers=MARKER_DICT,
        ax=ax[1],
    )
    ax[1].set_xlim(-0.12, 0.12)

    fig.tight_layout()
    plt.savefig(save_fn)
    plt.close()


def plot_c_index_difference(df, save_fn, sort_diff=False, pos_col='attention', neg_col='baseline'):
    style()
    diff = pd.DataFrame(
        df[pos_col] - df[neg_col],
        columns=[f"Difference in C-Index ({pos_col} - {neg_col})"],
        index=df.index,
    )
    if sort_diff:
        diff = diff.sort_values(by=diff.columns[0])
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        diff,
        x=diff.columns[0],
        y=diff.index,
        # hue="Model",
        s=100,
        # style="Model",
        # markers=MARKER_DICT,
    )
    fig.tight_layout()
    plt.xlim(-0.25, 0.25)

    plt.savefig(save_fn)
    plt.close()


def style():
    sns.set_theme(style="whitegrid")
    sns.set_style(
        {
            "axes.facecolor": "1",  # White axes background
            "grid.color": "0.9",  # Light grey grid lines
            "grid.linestyle": "-",  # Solid grid line
            "axes.grid.axis": "y",  # Grid lines along y-axis only
            "axes.grid": True,  # Enable grid
        }
    )


def plot_c_index_absolute(df, save_fn, add_mean_line=True, extra_title=None):
    """
    assumes each column is one model, each row is one target/disease. Index holds the target/disease names
    """
    style()
    df = df.copy()
    means = df.mean()
    df.fillna(0.0, inplace=True)
    if add_mean_line:
        df.loc["------"] = 0.0
        df.loc["mean"] = means
    df.columns = [f"{col} ({means[col]:.3f})" for col in df.columns]

    df = df.melt(
        var_name="Model",
        value_name="C-Index",
        ignore_index=False,
    )

    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        df,
        x="C-Index",
        y=df.index,
        hue="Model",
        s=100,
        # style="Model",
        # markers=MARKER_DICT,
    )
    # fig.tight_layout()
    plt.xlim(0.5, 1)
    # plt.legend(fontsize='x-small')
    plt.legend(loc="lower right", bbox_to_anchor=(0.85, 1), fontsize="x-small")
    if extra_title is not None:
        plt.title(extra_title)

    plt.savefig(save_fn)
    plt.close()
