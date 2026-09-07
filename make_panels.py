#!/usr/bin/env python3
"""Strip figures in the draft's house style: scatters, then bars with intervals.

Each figure is one row: the raw relationship for a few readouts, each titled
with its Spearman, then a bar panel putting every readout on one axis under a
single protocol, with group-bootstrap intervals and the measurement ceiling
drawn in.

Out-of-fold predictions are cached under <out>/cache so the figures can be
redrawn without refitting the wide probes.

    python3 make_panels.py --out results/figures
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_epistasis import gc_fraction, group_boot, out_of_fold_linear
from evo_probe import elements, kmers, out_of_fold
from single_variant import allele_features, single_variants

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
# Colour carries the kind of readout, not the series index.
COLOR = {"baseline": "#8a8a85", "likelihood": "#eb6834",
         "probe": "#2a78d6", "combined": "#1baf7a"}
POINT = "#3b6ea5"


def cached(cache, name, build):
    path = cache / f"{name}.npy"
    if path.exists():
        return np.load(path)
    value = build()
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    return value


def interval(prediction, y, groups, n_boot=1000, seed=0):
    rho = float(spearmanr(prediction, y).statistic)
    low, high = group_boot(groups, lambda i: spearmanr(prediction[i], y[i]).statistic,
                           n_boot, seed)
    return rho, low, high


def single_variant_panels(cache):
    evo = single_variants("results/evo2_7b_base/predictions.csv")
    ntv3 = single_variants("results/ntv3_100m_pre/predictions.csv")
    if not evo.sequence_id.equals(ntv3.sequence_id):
        raise SystemExit("the two single-variant tables are not aligned")
    y, groups = evo.y.to_numpy(), evo.group_id.to_numpy()
    seqs, refs = evo.seq.tolist(), evo.seq_ref.tolist()

    delta_kmers = cached(cache, "sv_delta_kmers",
                         lambda: kmers(seqs) - kmers(refs))

    def fit(x):
        # single_variant.py's protocol exactly: shuffled grouped folds, seed 0,
        # ridge only for multi-column features.  Using evo_probe's unshuffled
        # protocol here would put a different number on the same bar.
        return out_of_fold_linear(x, y, groups, 5, ridge=x.shape[1] > 1, seed=0)

    fits = {
        "delta k-mers": (cached(cache, "sv_pred_delta_kmers",
                                lambda: fit(delta_kmers)), "baseline"),
        "allele identity": (cached(cache, "sv_pred_allele",
                                   lambda: fit(allele_features(evo))), "baseline"),
        "GC content": (cached(cache, "sv_pred_gc",
                              lambda: fit(gc_fraction(seqs).reshape(-1, 1))), "baseline"),
        "Evo 2 likelihood": (cached(cache, "sv_pred_evo",
                                    lambda: fit(evo.delta_score.to_numpy().reshape(-1, 1))), "likelihood"),
        "NTv3 likelihood": (cached(cache, "sv_pred_ntv3",
                                   lambda: fit(ntv3.delta_score.to_numpy().reshape(-1, 1))), "likelihood"),
    }
    ceiling = json.loads(Path("results/evo2_7b_base/single_variant.json").read_text())["zero_shot"]["ceiling"]
    return {
        "y": y, "groups": groups, "n": len(evo), "ceiling": ceiling,
        "scatters": [
            ("Evo 2 delta log-likelihood", evo.delta_score.to_numpy(), "raw score change"),
            ("NTv3 delta pseudo-log-likelihood", ntv3.delta_score.to_numpy(), "raw score change"),
            ("delta k-mer counts, fitted", fits["delta k-mers"][0], "out-of-fold prediction"),
        ],
        "bars": {name: (pred, kind) for name, (pred, kind) in fits.items()},
        "ylabel": "measured effect of the substitution (log2)",
        "title": "Predicting single-variant effect from the same 200 bases",
    }


def element_panels(cache):
    el = elements()
    y, groups, seqs = el.activity.values, el.group.values, el.seq.values
    order = pd.Index(el.sequence_id)

    def probe_prediction(directory, pooling="mean"):
        """Reindex the saved embedding onto the element table's row order."""
        table = pd.read_csv(Path(directory) / "elements.csv")
        matrix = np.load(Path(directory) / f"X_{pooling}.npy")
        if len(table) != len(matrix):
            raise SystemExit(f"{directory}: {len(table)} rows but {len(matrix)} embeddings")
        position = {sequence: row for row, sequence in enumerate(table.sequence_id)}
        missing = [s for s in order if s not in position]
        if missing:
            raise SystemExit(f"{directory}: {len(missing)} elements have no embedding; re-run embed")
        return out_of_fold(matrix[[position[s] for s in order]], y, groups, 5)

    fits = {
        "GC content": (cached(cache, "el_pred_gc",
                              lambda: out_of_fold(gc_fraction(seqs).reshape(-1, 1), y, groups, 5)), "baseline"),
        "1/2/3-mer counts": (cached(cache, "el_pred_kmers",
                                    lambda: out_of_fold(kmers(seqs), y, groups, 5)), "baseline"),
        "Evo 2 likelihood": (cached(cache, "el_pred_evo_score",
                                    lambda: out_of_fold(el.s_wt.values.reshape(-1, 1), y, groups, 5)), "likelihood"),
        "Evo 2 probe": (cached(cache, "el_pred_evo_probe",
                               lambda: probe_prediction("results/evo_probe")), "probe"),
        "NTv3 100M probe": (cached(cache, "el_pred_ntv3_100m",
                                   lambda: probe_prediction("results/ntv3_100m_final")), "probe"),
        "NTv3 650M probe": (cached(cache, "el_pred_ntv3_650m",
                                   lambda: probe_prediction("results/ntv3_650m_final")), "probe"),
    }
    return {
        "y": y, "groups": groups, "n": len(el), "ceiling": None,
        "scatters": [
            ("Evo 2 probe, mean pooled", fits["Evo 2 probe"][0], "out-of-fold prediction"),
            ("1/2/3-mer counts, fitted", fits["1/2/3-mer counts"][0], "out-of-fold prediction"),
            ("Evo 2 log-likelihood", el.s_wt.values, "whole-sequence score"),
        ],
        "bars": {name: (pred, kind) for name, (pred, kind) in fits.items()},
        "ylabel": "measured reference activity (log2 RNA/DNA)",
        "title": "Predicting element activity from the same 200 bases",
    }


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, length=3, labelsize=8)
    axis.set_axisbelow(True)


def draw(panel, path):
    scatters, y, groups = panel["scatters"], panel["y"], panel["groups"]
    # The bar panel carries several multi-word labels, so give it more width
    # than a scatter rather than shrinking the type.
    figure, axes = plt.subplots(
        1, len(scatters) + 1,
        figsize=(3.35 * len(scatters) + 4.9, 4.0),
        gridspec_kw={"width_ratios": [1] * len(scatters) + [1.45]})
    figure.patch.set_facecolor("white")
    for axis, (label, x, xlabel) in zip(axes, scatters):
        axis.scatter(x, y, s=5, alpha=0.22, linewidths=0, color=POINT, rasterized=True)
        axis.set_title(f"{label}\nSpearman {spearmanr(x, y).statistic:+.3f}",
                       fontsize=9.5, color=INK)
        axis.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
        style(axis)
    axes[0].set_ylabel(panel["ylabel"], fontsize=9, color=INK)

    bars = axes[-1]
    scored = [(name, *interval(pred, y, groups), kind)
              for name, (pred, kind) in panel["bars"].items()]
    scored.sort(key=lambda r: -r[1])
    x = np.arange(len(scored))
    values = [r[1] for r in scored]
    lower = [r[1] - r[2] for r in scored]
    upper = [r[3] - r[1] for r in scored]
    bars.bar(x, values, color=[COLOR[r[4]] for r in scored], width=0.66, zorder=3)
    bars.errorbar(x, values, yerr=[lower, upper], fmt="none", ecolor=INK,
                  capsize=3.5, lw=1, zorder=4)
    bars.axhline(0, color=MUTED, lw=1, zorder=2)
    for index, (_, value, _, high, _) in enumerate(scored):
        bars.text(index, high + 0.018, f"{value:+.3f}", ha="center",
                  fontsize=8, color=INK)
    if panel["ceiling"]:
        bars.axhline(panel["ceiling"], color=MUTED, lw=1, ls=(0, (5, 4)), zorder=2)
        bars.text(len(scored) - 0.45, panel["ceiling"], f"ceiling {panel['ceiling']:.2f} ",
                  fontsize=8, color=MUTED, va="bottom", ha="right")
    bars.set_xticks(x, ["\n".join(textwrap.wrap(r[0], 11)) for r in scored],
                    fontsize=8, color=INK)
    bars.set_ylabel("out-of-fold Spearman", fontsize=9, color=INK)
    bars.set_title("Every readout, one protocol\ngrouped five-fold, 95% interval",
                   fontsize=9.5, color=INK)
    top = max(panel["ceiling"] or 0, max(r[3] for r in scored)) * 1.12
    bars.set_ylim(min(0, min(r[2] for r in scored)) * 1.35 - 0.02, top)
    bars.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    style(bars)

    figure.suptitle(f"{panel['title']}  (n = {panel['n']:,})", fontsize=11.5,
                    color=INK, x=0.008, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/figures", type=Path)
    args = parser.parse_args()
    cache = args.out / "cache"
    args.out.mkdir(parents=True, exist_ok=True)
    draw(single_variant_panels(cache), args.out / "panels_single_variant.png")
    draw(element_panels(cache), args.out / "panels_element_activity.png")


if __name__ == "__main__":
    main()
