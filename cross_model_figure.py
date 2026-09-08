#!/usr/bin/env python3
"""Plot standardized interaction measurements for every evaluated model.

    python3 cross_model_figure.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_RESULTS = (
    "results/evo2_7b_base",
    "results/ntv3_100m_pre",
    "results/ntv3_650m_pre",
)


def standardized(values):
    values = np.asarray(values, float)
    scale = values.std()
    if not np.isfinite(values).all() or scale == 0:
        raise ValueError("measurements must be finite and nonconstant")
    return (values - values.mean()) / scale


def load_result(directory):
    directory = Path(directory)
    frame = pd.read_csv(directory / "predictions.csv",
                        usecols=["pair_id", "epsilon", "model_interaction"])
    metrics = json.loads((directory / "metrics.json").read_text())
    if frame.pair_id.duplicated().any() or len(frame) != metrics["pairs"]:
        raise ValueError(f"{directory}: predictions do not contain one row per pair")
    if not np.isfinite(frame[["epsilon", "model_interaction"]].to_numpy(float)).all():
        raise ValueError(f"{directory}: predictions contain non-finite values")
    return metrics["label"], frame.set_index("pair_id"), metrics


def make_figure(result_dirs, output):
    loaded = [load_result(directory) for directory in result_dirs]
    reference_ids = loaded[0][1].index
    epsilon = loaded[0][1].epsilon
    for label, frame, _ in loaded[1:]:
        if not frame.index.equals(reference_ids) or not np.allclose(frame.epsilon, epsilon):
            raise ValueError(f"{label}: predictions do not use the shared experimental measurements")

    x = standardized(epsilon)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    correlations = []
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for axis, (label, frame, metrics), color in zip(axes.flat[:3], loaded, colors):
        y = standardized(frame.model_interaction)
        pearson = pearsonr(x, y).statistic
        spearman = spearmanr(x, y).statistic
        low, high = metrics["cluster_bootstrap_95ci"]["spearman"]
        axis.scatter(x, y, s=8, alpha=0.18, color=color, edgecolors="none")
        axis.axhline(0, color="0.75", linewidth=0.8)
        axis.axvline(0, color="0.75", linewidth=0.8)
        axis.set_title(label)
        axis.set_xlabel("Experimental epistasis (z-score)")
        axis.set_ylabel("Model interaction (z-score)")
        axis.text(0.03, 0.97,
                  f"Spearman $\\rho$ = {spearman:+.3f}\n"
                  f"95% CI [{low:+.3f}, {high:+.3f}]\n"
                  f"Pearson $r$ = {pearson:+.3f}",
                  transform=axis.transAxes, va="top",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        correlations.append((label, spearman, pearson))

    axis = axes.flat[3]
    labels = [row[0] for row in correlations]
    positions = np.arange(len(labels))
    width = 0.36
    axis.bar(positions - width / 2, [row[1] for row in correlations], width,
             label="Spearman $\\rho$", color="#4c78a8")
    axis.bar(positions + width / 2, [row[2] for row in correlations], width,
             label="Pearson $r$", color="#e45756")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Correlation with experimental epistasis")
    axis.set_title("Cross-model comparison")
    axis.legend(frameon=False)
    figure.suptitle("Standardized model interactions versus K562 MPRA epistasis (n = 2,833)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", default=DEFAULT_RESULTS,
                        help="result directories containing predictions.csv and metrics.json")
    parser.add_argument("--out", default="results/cross_model_standardized_correlation.png")
    args = parser.parse_args()
    correlations = make_figure(args.results, args.out)
    for label, spearman, pearson in correlations:
        print(f"{label}: Spearman {spearman:+.4f}; Pearson {pearson:+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
