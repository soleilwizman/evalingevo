#!/usr/bin/env python3
"""One chart: Spearman with the measured single-variant effect, eight readouts.

Four of the eight can be computed from what is committed. The other four
cannot, and the chart says so rather than leaving them out:

  NTv3 650M score   no scoring run exists; 650M was used for embeddings only
  every probe       the committed embeddings cover the 2,595 reference 200-mers
                    and none of the 5,428 variant sequences, so a variant-level
                    probe needs a GPU pass over the alternate sequences first

Protocol matches single_variant.py exactly: grouped five-fold shuffled with
seed 0, ridge for multi-column features, 95% interval resampling whole
overlapping-region groups.

    python3 single_variant_spearman.py --out results/figures/single_variant_spearman.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from evo_epistasis import gc_fraction, group_boot, out_of_fold_linear
from evo_probe import kmers
from single_variant import single_variants

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
COLOR = {"baseline": "#8a8a85", "likelihood": "#eb6834", "probe": "#2a78d6"}
KIND_LABEL = {"baseline": "sequence baseline (no model)",
              "likelihood": "model score, zero-shot",
              "probe": "model embedding probe, supervised"}
MISSING_PROBE = "no variant\nembeddings"
MISSING_SCORE = "no scoring\nrun"


def build(folds=5, seed=0, n_boot=1000):
    evo = single_variants("results/evo2_7b_base/predictions.csv")
    ntv3 = single_variants("results/ntv3_100m_pre/predictions.csv")
    if not evo.sequence_id.equals(ntv3.sequence_id):
        raise SystemExit("the two single-variant tables are not aligned")
    y, groups = evo.y.to_numpy(), evo.group_id.to_numpy()
    seqs, refs = evo.seq.tolist(), evo.seq_ref.tolist()

    available = [
        ("Evo 2 score", evo.delta_score.to_numpy().reshape(-1, 1), "likelihood"),
        ("NTv3 100M score", ntv3.delta_score.to_numpy().reshape(-1, 1), "likelihood"),
        ("GC content", gc_fraction(seqs).reshape(-1, 1), "baseline"),
        ("1/2/3-mer delta", kmers(seqs) - kmers(refs), "baseline"),
    ]
    scored = {}
    for label, features, kind in available:
        prediction = out_of_fold_linear(features, y, groups, folds,
                                        ridge=features.shape[1] > 1, seed=seed)
        rho = float(spearmanr(prediction, y).statistic)
        low, high = group_boot(groups, lambda i: spearmanr(prediction[i], y[i]).statistic,
                               n_boot, seed)
        scored[label] = {"label": label, "kind": kind, "spearman": rho,
                         "ci": [float(low), float(high)], "features": int(features.shape[1])}
        print(f"  {label:20} {rho:+.4f}  [{low:+.4f}, {high:+.4f}]", flush=True)

    order = [
        ("Evo 2 score", None), ("NTv3 100M score", None),
        ("NTv3 650M score", MISSING_SCORE),
        ("GC content", None), ("1/2/3-mer delta", None),
        ("NTv3 100M probe", MISSING_PROBE), ("NTv3 650M probe", MISSING_PROBE),
        ("Evo 2 probe", MISSING_PROBE),
    ]
    rows = []
    for label, missing in order:
        if missing:
            kind = "probe" if "probe" in label else "likelihood"
            rows.append({"label": label, "kind": kind, "spearman": None,
                         "ci": None, "not_computed": missing.replace("\n", " "),
                         "not_computed_display": missing})
            print(f"  {label:20} not computed: {missing.replace(chr(10), ' ')}")
        else:
            rows.append(scored[label])
    return {"n": int(len(evo)), "target": "measured log2 effect of one substitution",
            "protocol": "grouped five-fold shuffled seed 0; ridge for multi-column "
                        "features; 95% interval resamples whole region groups",
            "kmer_note": "the k-mer bar is the delta, counts in the alternate sequence "
                         "minus the reference, which for one substitution is the words "
                         "made and broken; absolute counts give -0.035",
            "readouts": rows}


def draw(data, path):
    rows = data["readouts"]
    x = np.arange(len(rows))
    done = [(i, r) for i, r in enumerate(rows) if r["spearman"] is not None]
    gaps = [(i, r) for i, r in enumerate(rows) if r["spearman"] is None]

    figure, axis = plt.subplots(figsize=(10.6, 5.8))
    figure.patch.set_facecolor("white")
    axis.bar([i for i, _ in done], [r["spearman"] for _, r in done],
             color=[COLOR[r["kind"]] for _, r in done], width=0.62, zorder=3)
    axis.errorbar([i for i, _ in done], [r["spearman"] for _, r in done],
                  yerr=[[r["spearman"] - r["ci"][0] for _, r in done],
                        [r["ci"][1] - r["spearman"] for _, r in done]],
                  fmt="none", ecolor=INK, capsize=4, lw=1.1, zorder=4)
    for i, r in done:
        axis.text(i, r["ci"][1] + 0.006, f"{r['spearman']:+.3f}",
                  ha="center", fontsize=9.5, color=INK)

    top = max(r["ci"][1] for _, r in done) + 0.045
    bottom = min(0, min(r["ci"][0] for _, r in done)) - 0.03
    for i, r in gaps:
        axis.add_patch(plt.Rectangle((i - 0.31, 0), 0.62, top * 0.42, facecolor="none",
                                     edgecolor=GRID, lw=1, ls=(0, (3, 3)), zorder=3))
        axis.text(i, top * 0.21, r["not_computed_display"], ha="center", va="center",
                  fontsize=8.5, color=MUTED, style="italic")
    axis.axhline(0, color=MUTED, lw=1, zorder=2)
    axis.set_xticks(x, [r["label"].replace(" ", "\n", 1) for r in rows],
                    fontsize=9.5, color=INK)
    for tick, row in zip(axis.get_xticklabels(), rows):
        tick.set_color(MUTED if row["spearman"] is None else INK)
    axis.set_ylabel("Spearman correlation with the measured single-variant effect",
                    fontsize=10, color=INK)
    axis.set_ylim(bottom, top)
    axis.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, length=3)
    axis.set_title(f"Predicting the effect of a single substitution  "
                   f"(n = {data['n']:,} variants)\n"
                   "out of fold, grouped five-fold, 95% interval over region groups",
                   fontsize=11.5, color=INK, loc="left", pad=12)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLOR[k], label=v)
               for k, v in KIND_LABEL.items()]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=9, color="white",
                              mec=GRID, label="not computed (needs a GPU pass)"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                  fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/figures/single_variant_spearman.png",
                        type=Path)
    args = parser.parse_args()
    data = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    draw(data, args.out)


if __name__ == "__main__":
    main()
