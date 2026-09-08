#!/usr/bin/env python3
"""Plot standardized single-variant and element measurements for evaluated models.

    python3 cross_model_activity_figure.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
                        usecols=["pair_id", "group_id", "seq_wt", "y_a", "y_b",
                                 "s_wt", "s_a", "s_b"])
    metrics = json.loads((directory / "metrics.json").read_text())
    if frame.pair_id.duplicated().any() or len(frame) != metrics["pairs"]:
        raise ValueError(f"{directory}: predictions do not contain one row per pair")
    if not np.isfinite(frame.drop(columns=["pair_id", "group_id", "seq_wt"]).to_numpy(float)).all():
        raise ValueError(f"{directory}: predictions contain non-finite values")
    return metrics["label"], frame, metrics


def element_measurements(frame, audit):
    key = ["v1", "v2", "center_variant", "window", "library"]
    audit_key = audit[key[0]].astype(str)
    for column in key[1:]:
        audit_key += ";" + audit[column].astype(str)
    activity = audit.assign(_key=audit_key).drop_duplicates("_key").set_index("_key").refref_Log2FC
    elements = frame.assign(_key=frame.pair_id.str.split("|").str[0])
    elements["activity"] = elements._key.map(activity)
    elements = elements.dropna(subset=["activity"]).drop_duplicates("seq_wt")
    if elements.seq_wt.duplicated().any() or elements.group_id.nunique() == 0:
        raise ValueError("element measurements are not unique and grouped")
    return elements.activity.to_numpy(float), elements.s_wt.to_numpy(float)


def make_figure(result_dirs, output, audit_path="data/audit.csv.gz"):
    loaded = [load_result(directory) for directory in result_dirs]
    audit = pd.read_csv(audit_path, usecols=["v1", "v2", "center_variant", "window", "library", "refref_Log2FC"])
    reference = loaded[0][1]
    for label, frame, _ in loaded[1:]:
        if not frame.pair_id.equals(reference.pair_id):
            raise ValueError(f"{label}: predictions do not use the same variant pairs")
        if not np.allclose(frame[["y_a", "y_b"]], reference[["y_a", "y_b"]]):
            raise ValueError(f"{label}: predictions do not use the shared experimental measurements")

    figure, axes = plt.subplots(2, len(loaded), figsize=(4.4 * len(loaded), 8), constrained_layout=True,
                               sharex="row", sharey="row")
    correlations = {"single_variant_magnitude": [], "element_activity": []}
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for column, ((label, frame, metrics), color) in enumerate(zip(loaded, colors)):
        single_y = np.abs(np.r_[frame.y_a, frame.y_b])
        single_score = np.abs(np.r_[frame.s_a - frame.s_wt, frame.s_b - frame.s_wt])
        element_y, element_score = element_measurements(frame, audit)
        panels = (
            (axes[0, column], single_y, single_score, "Single-variant effect magnitude",
             "Experimental |effect| (z-score)", "Model |score change| (z-score)",
             metrics["audit_analyses"]["single_variants"]["magnitude_95ci"]),
            (axes[1, column], element_y, element_score, "Reference-element activity",
             "Experimental activity (z-score)", "Model sequence score (z-score)",
             metrics["audit_analyses"]["elements"]["raw_spearman"]["model_95ci"]),
        )
        for axis, measurement, score, title, xlabel, ylabel, interval in panels:
            rho = spearmanr(measurement, score).statistic
            axis.scatter(standardized(measurement), standardized(score), s=8, alpha=0.18,
                         color=color, edgecolors="none")
            axis.axhline(0, color="0.75", linewidth=0.8)
            axis.axvline(0, color="0.75", linewidth=0.8)
            axis.set_title(f"{label}\n{title}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:+.3f}\n95% CI [{interval[0]:+.3f}, {interval[1]:+.3f}]",
                      transform=axis.transAxes, va="top",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
            correlations["single_variant_magnitude" if axis is axes[0, column] else "element_activity"].append((label, rho))
    figure.suptitle("Cross-model comparison with standardized K562 MPRA measurements")
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
    parser.add_argument("--audit", default="data/audit.csv.gz")
    parser.add_argument("--out", default="results/cross_model_variant_and_element.png")
    args = parser.parse_args()
    correlations = make_figure(args.results, args.out, args.audit)
    for level, rows in correlations.items():
        for label, spearman in rows:
            print(f"{level}; {label}: Spearman {spearman:+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
