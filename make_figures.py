#!/usr/bin/env python3
"""Build the benchmark figures from the committed results.

Two figures, both read straight off the files under results/ so the picture and
the numbers cannot drift apart:

  benchmark.png            out-of-fold Spearman for every readout on each task,
                           model likelihoods and embedding probes against the
                           cheap baselines they have to beat
  margin_over_baseline.png the same comparison stated as the claim actually
                           being made: readout minus the strongest baseline,
                           with its group-bootstrap interval and a zero line

    python3 make_figures.py --out results/figures

Element-level numbers come from the probe.txt files, which share one protocol
(grouped five-fold, folds unshuffled, RidgeCV inside each training fold).
Single-variant numbers come from single_variant.json, same folds but shuffled
with seed 0. The two protocols agree to about 0.006 on the shared k-mer
baseline, so panels are compared within a task, never across at the third
decimal.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Roles, not decoration: gray is the reference material, one hue per kind of
# model readout.  Validated as a categorical set against a light surface;
# the aqua slot sits below 3:1 so every bar carries a direct value label.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
COLOR = {"baseline": "#8a8a85", "likelihood": "#eb6834",
         "probe": "#2a78d6", "combined": "#1baf7a"}
KIND_LABEL = {"baseline": "sequence baseline (no model)",
              "likelihood": "model likelihood, zero-shot",
              "probe": "model embedding probe, supervised",
              "combined": "model readout plus the baseline"}


def probe_value(path, label):
    """Pull one row out of a probe.txt table, or say exactly what was missing."""
    text = Path(path).read_text()
    match = re.search(rf"^{re.escape(label)}\s+([+-][\d.]+)\s", text, re.M)
    if not match:
        raise SystemExit(f"{path}: no row '{label}'. Re-run the probe; the file "
                         "may be truncated.")
    return float(match.group(1))


def probe_margin(path):
    match = re.search(r"probe minus (?:word counts|k-mers): ([+-][\d.]+)\s+"
                      r"95% interval \[([+-][\d.]+), ([+-][\d.]+)\]", Path(path).read_text())
    if not match:
        raise SystemExit(f"{path}: no 'probe minus' line. The file is truncated.")
    return tuple(float(g) for g in match.groups())


def element_ceiling(predictions, audit):
    """Reliability of the measured element activity, same diagnostic used elsewhere."""
    pred, table = pd.read_csv(predictions), pd.read_csv(audit)
    key = ["v1", "v2", "center_variant", "window", "library"]
    joined = table[key[0]].astype(str)
    for column in key[1:]:
        joined = joined + ";" + table[column].astype(str)
    merged = pred.assign(_k=pred.pair_id.str.split("|").str[0]).merge(
        table.assign(_k=joined).drop_duplicates("_k")[["_k", "refref_Log2FC", "refref_Log2FC_SE"]],
        on="_k", how="left")
    el = merged.dropna(subset=["refref_Log2FC"]).drop_duplicates("seq_wt")
    y, se = el.refref_Log2FC.to_numpy(float), el.refref_Log2FC_SE.to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(se)
    y, se = y[keep], se[keep]
    return float(np.sqrt(max(1 - (se ** 2).mean() / y.var(ddof=0), 0.0))), int(len(y))


def collect(results):
    """Every number the figures draw, with its source file."""
    evo = results / "evo_probe/probe.txt"
    evo_last = results / "evo_probe/probe_last.txt"
    n100 = results / "ntv3_100m_final/probe.txt"
    n650 = results / "ntv3_650m_final/probe.txt"
    kmer = probe_value(evo, "DNA word counts (84 features)")
    element = [
        ("Evo 2 log-likelihood", probe_value(evo, "Evo score (1 feature)"), "likelihood"),
        ("GC content", probe_value(evo, "GC content (1 feature)"), "baseline"),
        ("Evo 2 probe, last token", probe_value(evo_last, "Evo hidden layer blocks.26.mlp.l3 (probe)"), "probe"),
        ("NTv3 650M probe", probe_value(n650, "NTv3 hidden layer (probe)"), "probe"),
        ("NTv3 100M probe", probe_value(n100, "NTv3 hidden layer (probe)"), "probe"),
        ("1/2/3-mer counts", kmer, "baseline"),
        ("NTv3 650M probe + k-mers", probe_value(n650, "NTv3 hidden layer + word counts"), "combined"),
        ("NTv3 100M probe + k-mers", probe_value(n100, "NTv3 hidden layer + word counts"), "combined"),
        ("Evo 2 probe, mean pooled", probe_value(evo, "Evo hidden layer blocks.26.mlp.l3 (probe)"), "probe"),
    ]
    element_margins = [
        ("Evo 2 probe, mean pooled", *probe_margin(evo), "probe"),
        ("Evo 2 probe, last token", *probe_margin(evo_last), "probe"),
        ("NTv3 100M probe", *probe_margin(n100), "probe"),
        ("NTv3 650M probe", *probe_margin(n650), "probe"),
    ]

    evo_sv = json.loads((results / "evo2_7b_base/single_variant.json").read_text())
    ntv3_sv = json.loads((results / "ntv3_100m_pre/single_variant.json").read_text())
    e, n = evo_sv["supervised_out_of_fold"], ntv3_sv["supervised_out_of_fold"]
    single = [
        ("variant position", e["position"]["spearman"], "baseline"),
        ("1/2/3-mer counts, absolute", e["kmer_counts"]["spearman"], "baseline"),
        ("GC content", e["gc_content"]["spearman"], "baseline"),
        ("Evo 2 delta log-likelihood", e["model_delta_score"]["spearman"], "likelihood"),
        ("NTv3 delta pseudo-log-likelihood", n["model_delta_score"]["spearman"], "likelihood"),
        ("ref/alt allele identity", e["allele_identity"]["spearman"], "baseline"),
        ("delta k-mers + Evo 2", e["kmer_delta_plus_model"]["spearman"], "combined"),
        ("delta k-mers + NTv3", n["kmer_delta_plus_model"]["spearman"], "combined"),
        ("delta k-mers, words made and broken", e["kmer_delta"]["spearman"], "baseline"),
    ]
    single_margins = []
    for label, source, key, kind in [
            ("Evo 2 delta log-likelihood", e, "model_delta_score", "likelihood"),
            ("NTv3 delta pseudo-log-likelihood", n, "model_delta_score", "likelihood"),
            ("delta k-mers + Evo 2", e, "kmer_delta_plus_model", "combined"),
            ("delta k-mers + NTv3", n, "kmer_delta_plus_model", "combined")]:
        entry = source["margin_over_kmer_delta"][key]
        single_margins.append((label, entry["margin"], *entry["interval"], kind))

    ceiling, n_elements = element_ceiling(results / "evo2_7b_base/predictions.csv",
                                          Path("data/audit.csv.gz"))
    return {
        "element": {"rows": element, "margins": element_margins, "baseline": kmer,
                    "baseline_name": "1/2/3-mer counts", "ceiling": ceiling, "n": n_elements,
                    "title": "Predicting element activity",
                    "subtitle": "one value per distinct reference 200-mer"},
        "single": {"rows": single, "margins": single_margins,
                   "baseline": e["kmer_delta"]["spearman"],
                   "baseline_name": "delta k-mers",
                   "ceiling": evo_sv["zero_shot"]["ceiling"], "n": evo_sv["zero_shot"]["n"],
                   "title": "Predicting single-variant effect",
                   "subtitle": "one value per substitution, ref/alt coding"},
    }


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, length=3)
    axis.set_axisbelow(True)


def bar_panel(axis, panel, note=None):
    rows = sorted(panel["rows"], key=lambda r: r[1])
    names = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])
    colors = [COLOR[r[2]] for r in rows]
    y = np.arange(len(rows))
    axis.barh(y, values, color=colors, height=0.66, zorder=3)
    axis.axvline(0, color=MUTED, lw=1, zorder=2)
    axis.axvline(panel["baseline"], color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
    axis.set_yticks(y, names, fontsize=9, color=INK)
    axis.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    span = values.max() - min(values.min(), 0)
    pad = span * 0.05
    # White pad behind each value label so it stays legible where it crosses the
    # baseline rule.
    for index, value in enumerate(values):
        axis.text(value + (pad * 0.4 if value >= 0 else -pad * 0.4), index, f"{value:+.3f}",
                  va="center", ha="left" if value >= 0 else "right", fontsize=8.5, color=INK,
                  zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    left = min(values.min() - pad * 3, -pad)
    axis.set_xlim(left, values.max() + pad * 7)
    # Anchor the baseline caption on whichever side of the rule has room.
    near_right = panel["baseline"] > left + 0.62 * (values.max() + pad * 7 - left)
    axis.text(panel["baseline"], len(rows) - 0.15,
              f"baseline to beat: {panel['baseline_name']}  " if near_right
              else f"  baseline to beat: {panel['baseline_name']}",
              fontsize=8.5, color=INK, style="italic", va="center",
              ha="right" if near_right else "left")
    best = values.max()
    axis.set_title(
        f"{panel['title']}   (n = {panel['n']:,}, {panel['subtitle']})\n"
        f"measurement caps any predictor at {panel['ceiling']:.3f}; "
        f"the best readout here reaches {best:.3f}, "
        f"{100 * best / panel['ceiling']:.0f}% of that",
        fontsize=10.5, color=INK, loc="left", pad=10)
    if note:
        axis.text(0.38, 0.03, note, transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=8.5, color=MUTED, style="italic")
    style(axis)


def margin_panel(axis, rows, title):
    rows = sorted(rows, key=lambda r: r[1])
    y = np.arange(len(rows))
    for index, (_, margin, low, high, kind) in enumerate(rows):
        axis.plot([low, high], [index, index], color=COLOR[kind], lw=2.2,
                  solid_capstyle="round", zorder=3)
        axis.plot([margin], [index], "o", color=COLOR[kind], ms=7,
                  mec="white", mew=1.5, zorder=4)
        axis.text(high + (high - low) * 0.06, index,
                  f"{margin:+.3f}  [{low:+.3f}, {high:+.3f}]",
                  va="center", fontsize=8.5, color=INK, zorder=6,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    axis.axvline(0, color=INK, lw=1.2, zorder=2)
    axis.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
    axis.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    # Always keep zero in frame: the whole claim is which side of it a row sits on.
    lo = min([r[2] for r in rows] + [0.0])
    hi = max([r[3] for r in rows] + [0.0])
    axis.set_xlim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.85)
    axis.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    style(axis)


def legend(figure, kinds):
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLOR[k],
                          label=KIND_LABEL[k]) for k in kinds]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                  fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--out", default="results/figures", type=Path)
    args = parser.parse_args()
    data = collect(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(11.4, 9.4))
    figure.patch.set_facecolor("white")
    bar_panel(axes[0], data["element"])
    bar_panel(axes[1], data["single"],
              note="no embedding probe on this task yet: the committed\nembeddings cover reference sequences only")
    for axis in axes:
        axis.set_xlabel("out-of-fold Spearman correlation with the measurement",
                        fontsize=9, color=MUTED)
    figure.suptitle("Frozen genomic language models against cheap sequence baselines",
                    fontsize=13, color=INK, x=0.012, ha="left", y=0.995)
    legend(figure, ["baseline", "likelihood", "probe", "combined"])
    figure.tight_layout(rect=(0, 0.045, 1, 0.975))
    first = args.out / "benchmark.png"
    figure.savefig(first, dpi=200, facecolor="white")
    plt.close(figure)

    # Three questions, three scales.  Putting a -0.19 margin and a -0.001 margin
    # on one axis would flatten the second to nothing, and the second is the one
    # that asks whether the model adds anything at all.
    alone = [r for r in data["single"]["margins"] if r[4] == "likelihood"]
    combined = [r for r in data["single"]["margins"] if r[4] == "combined"]
    figure, axes = plt.subplots(3, 1, figsize=(11.4, 7.6),
                                gridspec_kw={"height_ratios": [4, 2, 2]})
    figure.patch.set_facecolor("white")
    margin_panel(axes[0], data["element"]["margins"],
                 "Element activity: each probe minus 1/2/3-mer counts")
    margin_panel(axes[1], alone,
                 "Single-variant effect: the model's likelihood alone minus delta k-mers")
    margin_panel(axes[2], combined,
                 "Single-variant effect: delta k-mers plus the model, minus delta k-mers alone")
    for axis in axes:
        axis.set_xlabel("difference in out-of-fold Spearman, 95% interval over region groups",
                        fontsize=9, color=MUTED)
    figure.suptitle("Does the model beat the baseline? Anything crossing zero does not.",
                    fontsize=13, color=INK, x=0.012, ha="left", y=0.995)
    legend(figure, ["likelihood", "probe", "combined"])
    figure.tight_layout(rect=(0, 0.055, 1, 0.965))
    second = args.out / "margin_over_baseline.png"
    figure.savefig(second, dpi=200, facecolor="white")
    plt.close(figure)

    (args.out / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {first}\nwrote {second}\nwrote {args.out / 'figure_data.json'}")


if __name__ == "__main__":
    main()
