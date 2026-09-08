"""One figure comparing every model we have scored against the same measurements.

Only same-protocol numbers share a panel.  The three score-based rows come from
``evaluate`` (grouped 5-fold, shuffled with --seed, cluster-bootstrap intervals over
overlap groups).  The embedding row comes from the probe scripts, which use the same
grouped folds without shuffling, so it sits in its own panel and is reported the way
CLAUDE.md requires: the margin over 1/2/3-mer counts with its paired interval, not the
absolute level.

Run:  python3 model_comparison.py --out results/model_comparison.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-3 of the validated palette; entity, never rank
COLOR = {"Evo 2 7B base": "#2a78d6", "NTv3 100M pre": "#eb6834", "NTv3 650M pre": "#1baf7a"}
REFERENCE = "#7a7975"
SURFACE, INK, INK_SOFT = "#fcfcfb", "#0b0b0b", "#52514e"

SCORE_RUNS = (("Evo 2 7B base", "results/evo2_7b_base/metrics.json"),
              ("NTv3 100M pre", "results/ntv3_100m_pre/metrics.json"))
PROBE_RUNS = (("Evo 2 7B base", "results/evo_probe/probe.txt", "blocks.26.mlp.l3"),
              ("NTv3 100M pre", "results/ntv3_100m_final/probe.txt", "block 5"),
              ("NTv3 650M pre", "results/ntv3_650m_final/probe.txt", "block 11"))


def read_probe(path):
    """Pull the probe's own Spearman and its paired margin over word counts."""
    text = Path(path).read_text()
    level = re.search(r"^\S.*hidden layer \(probe\)\s+([+-][\d.]+)", text, re.M)
    margin = re.search(r"probe minus word counts:\s*([+-][\d.]+)\s*"
                       r"95% interval \[([+-][\d.]+),\s*([+-][\d.]+)\]", text)
    if not level or not margin:
        raise SystemExit(f"{path} has no probe row or paired interval; re-run the probe")
    return float(level.group(1)), tuple(float(margin.group(i)) for i in (1, 2, 3))


def collect():
    panels = []
    metrics = {name: json.loads(Path(p).read_text()) for name, p in SCORE_RUNS}

    rows = [(name, COLOR[name], m["raw"]["spearman"], m["cluster_bootstrap_95ci"]["spearman"])
            for name, m in metrics.items()]
    panels.append({"title": "Two-variant interaction", "rows": rows,
                   "note": "model contrast vs measured epistasis, recoded coding, "
                           f"n = {metrics['Evo 2 7B base']['pairs']:,} pairs",
                   "xlabel": "Spearman (cluster-bootstrap 95% CI)"})

    rows = [(name, COLOR[name], m["audit_analyses"]["single_variants"]["magnitude_spearman"],
             m["audit_analyses"]["single_variants"]["magnitude_95ci"]) for name, m in metrics.items()]
    panels.append({"title": "Single-variant effect size", "rows": rows,
                   "note": "|score change| vs |activity change|, "
                           f"n = {metrics['Evo 2 7B base']['audit_analyses']['single_variants']['n']:,} variants",
                   "xlabel": "Spearman (cluster-bootstrap 95% CI)"})

    raw = {name: m["audit_analyses"]["elements"]["raw_spearman"] for name, m in metrics.items()}
    rows = [(name, COLOR[name], r["model_vs_activity"], r["model_95ci"]) for name, r in raw.items()]
    gc = raw["Evo 2 7B base"]
    rows.append(("GC fraction (reference)", REFERENCE, gc["gc_vs_activity"], gc["gc_95ci"]))
    panels.append({"title": "Element activity, zero-shot likelihood", "rows": rows,
                   "note": "whole-sequence score vs reference activity, "
                           f"n = {metrics['Evo 2 7B base']['audit_analyses']['elements']['n_elements']:,} elements",
                   "xlabel": "Spearman (cluster-bootstrap 95% CI)"})

    rows = []
    for name, path, layer in PROBE_RUNS:
        level, (margin, low, high) = read_probe(path)
        rows.append((f"{name}\n{layer}, ρ = {level:+.3f}", COLOR[name], margin, [low, high]))
    panels.append({"title": "Element activity, learned probe on embeddings", "rows": rows,
                   "note": "advantage over 1/2/3-mer counts (ρ = +0.456), n = 2,595 elements",
                   "xlabel": "Spearman margin over word counts (paired 95% CI)"})
    return panels


def draw_panel(axis, panel):
    rows = panel["rows"]
    positions = range(len(rows) - 1, -1, -1)
    for y, (label, color, value, (low, high)) in zip(positions, rows):
        axis.plot([low, high], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=2)
        axis.plot([value], [y], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                  markeredgewidth=2, zorder=3)
        axis.text(1.02, y, f"{value:+.4f}", transform=axis.get_yaxis_transform(),
                  va="center", ha="left", fontsize=9, color=INK)
    axis.axvline(0.0, color="#b8b7b2", lw=1.0, zorder=1)
    axis.set_yticks(list(positions))
    axis.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    axis.set_ylim(-0.65, len(rows) - 0.35)
    axis.set_xlabel(panel["xlabel"], fontsize=9, color=INK_SOFT)
    axis.set_title(panel["title"], fontsize=11.5, color=INK, loc="left", pad=16)
    axis.text(0.0, 1.015, panel["note"], transform=axis.transAxes, fontsize=8.5,
              color=INK_SOFT, va="bottom")
    axis.grid(axis="x", color="#e6e5e0", lw=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=8.5, colors=INK_SOFT, length=3)
    axis.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#d7d6d1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/model_comparison.png")
    args = parser.parse_args()

    panels = collect()
    # row heights track row counts so a two-row panel is not stretched to fill a three-row box
    heights = [max(len(p["rows"]) for p in panels[:2]), max(len(p["rows"]) for p in panels[2:])]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), facecolor=SURFACE,
                                gridspec_kw={"height_ratios": heights})
    for axis, panel in zip(axes.ravel(), panels):
        axis.set_facecolor(SURFACE)
        draw_panel(axis, panel)

    handles = [plt.Line2D([], [], color=c, lw=2.0, marker="o", markersize=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, label=n)
               for n, c in COLOR.items()]
    handles.append(plt.Line2D([], [], color=REFERENCE, lw=2.0, marker="o", markersize=7,
                              markeredgecolor=SURFACE, markeredgewidth=1.5, label="sequence-only reference"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                  fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, -0.012))
    figure.suptitle("Frozen genomic language models against the Siraj et al. K562 MPRA",
                    fontsize=13.5, color=INK, x=0.008, ha="left", y=0.995)
    figure.text(0.008, 0.938,
                "An interval crossing zero is no measured association. Panels share measurements, "
                "not protocols: the top three come from evaluate, the bottom right from the probe scripts.",
                fontsize=9, color=INK_SOFT, ha="left")
    figure.tight_layout(rect=(0.0, 0.05, 0.955, 0.915), h_pad=3.0, w_pad=7.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")
    for panel in panels:
        print(f"\n{panel['title']}  ({panel['note']})")
        for label, _, value, (low, high) in panel["rows"]:
            print(f"  {label.replace(chr(10), '  '):<44} {value:+.4f}  [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
