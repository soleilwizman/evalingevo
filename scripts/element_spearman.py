#!/usr/bin/env python3
"""One chart: Spearman with measured element activity for five readouts.

Evo 2 score, NTv3 100M score, GC content, 1/2/3-mer counts, NTv3 100M probe.
Every bar is fitted out of fold under one protocol, grouped five-fold on the
overlapping-region groups, with a 95% interval from resampling whole groups.
The two model scores are single features, so fitting them out of fold only
sets their scale and sign; it cannot manufacture a correlation.

    python3 scripts/element_spearman.py --out results/figures/element_spearman.png
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

from evo_epistasis import gc_fraction, group_boot
from evo_probe import elements, kmers, out_of_fold

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
COLOR = {"baseline": "#8a8a85", "likelihood": "#eb6834", "probe": "#2a78d6"}
KIND_LABEL = {"baseline": "sequence baseline (no model)",
              "likelihood": "model score, zero-shot",
              "probe": "model embedding probe, supervised"}


def build(evo_predictions, ntv3_predictions, probe_dir, folds=5, n_boot=1000, seed=0):
    evo = elements(evo_predictions)
    ntv3 = elements(ntv3_predictions)
    if not evo.sequence_id.equals(ntv3.sequence_id):
        raise SystemExit("the two element tables are not aligned")
    y, groups, seqs = evo.activity.values, evo.group.values, evo.seq.values

    table = pd.read_csv(Path(probe_dir) / "elements.csv")
    matrix = np.load(Path(probe_dir) / "X_mean.npy")
    position = {s: i for i, s in enumerate(table.sequence_id)}
    missing = [s for s in evo.sequence_id if s not in position]
    if missing:
        raise SystemExit(f"{probe_dir}: {len(missing)} elements have no embedding")
    probe = matrix[[position[s] for s in evo.sequence_id]]

    readouts = [
        ("Evo 2 score", evo.s_wt.values.reshape(-1, 1), "likelihood"),
        ("NTv3 100M score", ntv3.s_wt.values.reshape(-1, 1), "likelihood"),
        ("GC content", gc_fraction(seqs).reshape(-1, 1), "baseline"),
        ("1/2/3-mer counts", kmers(seqs), "baseline"),
        ("NTv3 100M probe", probe, "probe"),
    ]
    rows = []
    for label, features, kind in readouts:
        prediction = out_of_fold(features, y, groups, folds)
        rho = float(spearmanr(prediction, y).statistic)
        low, high = group_boot(groups, lambda i: spearmanr(prediction[i], y[i]).statistic,
                              n_boot, seed)
        rows.append({"label": label, "kind": kind, "spearman": rho,
                     "ci": [float(low), float(high)], "features": int(features.shape[1])})
        print(f"  {label:20} {rho:+.4f}  [{low:+.4f}, {high:+.4f}]", flush=True)
    return {"n": int(len(evo)), "folds": folds, "bootstrap": n_boot,
            "protocol": "grouped five-fold, folds unshuffled, RidgeCV inside each "
                        "training fold; 95% interval resamples whole region groups",
            "readouts": rows}


def draw(data, path):
    rows = sorted(data["readouts"], key=lambda r: -r["spearman"])
    x = np.arange(len(rows))
    values = [r["spearman"] for r in rows]
    lower = [r["spearman"] - r["ci"][0] for r in rows]
    upper = [r["ci"][1] - r["spearman"] for r in rows]

    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    figure.patch.set_facecolor("white")
    axis.bar(x, values, color=[COLOR[r["kind"]] for r in rows], width=0.62, zorder=3)
    axis.errorbar(x, values, yerr=[lower, upper], fmt="none", ecolor=INK,
                  capsize=4, lw=1.1, zorder=4)
    axis.axhline(0, color=MUTED, lw=1, zorder=2)
    for index, row in enumerate(rows):
        axis.text(index, row["ci"][1] + 0.014, f"{row['spearman']:+.3f}",
                  ha="center", fontsize=9.5, color=INK)
    axis.set_xticks(x, [r["label"].replace(" ", "\n", 1) for r in rows],
                    fontsize=9.5, color=INK)
    axis.set_ylabel("Spearman correlation with measured element activity",
                    fontsize=10, color=INK)
    axis.set_ylim(min(0, min(r["ci"][0] for r in rows)) - 0.05,
                  max(r["ci"][1] for r in rows) + 0.075)
    axis.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, length=3)
    axis.set_title(f"Predicting element activity from the same 200 bases  "
                   f"(n = {data['n']:,} elements)\n"
                   "out of fold, grouped five-fold, 95% interval over region groups",
                   fontsize=11.5, color=INK, loc="left", pad=12)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLOR[k], label=v)
               for k, v in KIND_LABEL.items()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evo", default="results/evo2_7b_base/predictions.csv")
    parser.add_argument("--ntv3", default="results/ntv3_100m_pre/predictions.csv")
    parser.add_argument("--probe", default="results/ntv3_100m_final")
    parser.add_argument("--out", default="results/figures/element_spearman.png", type=Path)
    args = parser.parse_args()
    data = build(args.evo, args.ntv3, args.probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    draw(data, args.out)


if __name__ == "__main__":
    main()
