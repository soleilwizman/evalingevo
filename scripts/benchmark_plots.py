"""Plots for quartet and element evaluation reports."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark_data import CONTRAST
from benchmark_stats import correlations, group_boot
from scipy.stats import spearmanr


def make_plot(df, path, label):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].scatter(df.model_interaction, df.epsilon, s=8, alpha=0.4)
    axes[0, 0].set(
        xlabel=f"{label} interaction",
        ylabel="Measured epistasis",
        title="Raw association",
    )
    axes[0, 1].scatter(df.calibrated_model, df.epsilon, s=8, alpha=0.4)
    lo, hi = (
        min(df.calibrated_model.min(), df.epsilon.min()),
        max(df.calibrated_model.max(), df.epsilon.max()),
    )
    axes[0, 1].plot([lo, hi], [lo, hi], color="grey")
    axes[0, 1].set(xlabel="OOF prediction", ylabel="Measured epistasis", title="Calibrated")
    axes[0, 2].hist(df.epsilon, bins=40)
    axes[0, 2].set(title="Measured epistasis", xlabel="log2 activity")
    axes[1, 0].hist(df.model_interaction, bins=40)
    axes[1, 0].set(title="Model interaction", xlabel="score units")
    bins = (
        ("distance", pd.cut(df.distance, [0, 10, 25, 50, 100, np.inf])),
        (
            "|epistasis|",
            pd.cut(df.epsilon.abs(), [0, 0.25, 0.5, 1, np.inf], include_lowest=True),
        ),
    )
    strata = []
    for ax, (name, groups) in zip(axes[1, 1:], bins):
        values = (
            df.assign(bin=groups)
            .groupby("bin", observed=True)
            .apply(
                lambda x: pd.Series(
                    {
                        "n": len(x),
                        "mae": np.mean(np.abs(x.epsilon - x.calibrated_model)),
                        **correlations(x.epsilon, x.model_interaction),
                    }
                ),
                include_groups=False,
            )
        )
        ax.bar(values.index.astype(str), values.mae)
        ax.tick_params(axis="x", rotation=25)
        ax.set(title=f"Error by {name}", ylabel="OOF MAE")
        strata.extend(
            {
                "stratification": name,
                "bin": str(i),
                "n": int(r.n),
                "mae": float(r.mae),
                "spearman": None if pd.isna(r.spearman) else float(r.spearman),
                "pearson": None if pd.isna(r.pearson) else float(r.pearson),
            }
            for i, r in values.iterrows()
        )
    fig.suptitle(f"{label}: {CONTRAST}")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return strata


def make_element_plot(
    elements,
    y,
    gc,
    model,
    predictions,
    groups,
    path,
    n_boot=1000,
    seed=0,
    label="Evo 2",
):
    """Element-level baselines: GC and k-mer counts against the model likelihood."""
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    panels = (
        (gc, "GC fraction of the 200-mer", "GC content"),
        (predictions["kmer_1_2_3"], "out-of-fold prediction", "1/2/3-mer counts"),
        (model, f"{label} log-likelihood", f"{label}\nlikelihood"),
    )
    for axis, (x, xlabel, title) in zip(axes, panels):
        axis.scatter(x, y, s=6, alpha=0.25, linewidths=0, color="#3b6ea5")
        axis.set_xlabel(xlabel)
        axis.set_title(f"{title}\nSpearman {spearmanr(x, y).statistic:+.3f}")
    axes[0].set_ylabel("measured reference activity (log2 RNA/DNA)")

    order = ["kmer_1_2_3", "gc", "model"]
    labels = ["1/2/3-mer\ncounts", "GC\ncontent", f"{label}\nlikelihood"]
    values, lows, highs = [], [], []
    for name in order:
        prediction = predictions[name]
        rho = float(spearmanr(prediction, y).statistic)
        low, high = group_boot(
            groups, lambda i: spearmanr(prediction[i], y[i]).statistic, n_boot, seed
        )
        values.append(rho)
        lows.append(rho - low)
        highs.append(high - rho)
    axes[3].bar(labels, values, color=["#2a7f62", "#3b6ea5", "#b4483c"], width=0.6)
    axes[3].errorbar(
        labels, values, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4, lw=1
    )
    axes[3].axhline(0, color="black", lw=0.8)
    for i, value in enumerate(values):
        axes[3].text(i, value + highs[i] + 0.015, f"{value:+.3f}", ha="center", fontsize=9)
    axes[3].set_ylabel("out-of-fold Spearman")
    axes[3].set_title("grouped 5-fold, same protocol\nfor every row")
    axes[3].set_ylim(min(0, min(values)) - 0.05, max(values) + 0.09)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{label}: predicting enhancer activity from the same 200 bases (n = {len(y)} elements)",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path)
