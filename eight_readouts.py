#!/usr/bin/env python3
"""Two panels, the same eight readouts on each: element activity, then the
single-variant effect.

  Evo 2 zero-shot, NTv3 650M zero-shot, DNABERT-2 zero-shot,
  GC content, 1/2/3-mer counts,
  DNABERT-2 probe (block 11 of 12), NTv3 650M probe, Evo 2 probe (blocks.26.mlp.l3)

Each panel is fitted under one protocol, the one its existing scripts use:

  element  element_spearman.py: grouped five-fold, folds unshuffled, RidgeCV in
           each training fold (evo_probe.out_of_fold), n = 2,595
  variant  single_variant_spearman.py / variant_probe_fit.py: grouped five-fold
           shuffled with seed 0, ridge for multi-column features
           (evo_epistasis.out_of_fold_linear), n = 5,666

Zero-shot rows are the raw Spearman of the model's own score, nothing fitted.
Every interval resamples whole overlapping-region groups.

What cannot be drawn from the committed results is drawn as a gap, not left out:

  DNABERT-2 zero-shot   drawn once results/dnabert2_117m/predictions.csv exists
                        (dnabert2_score.py, then evaluate); a gap until then
  NTv3 650M variant probe at the deconv layer
                        the committed variant embeddings are the transformer
                        bottleneck (block 11); the bar shows that, labelled
  Evo 2 variant probe   embeddings never committed; carried from
                        results/vp_evo2/probe.txt, hatched, no interval

    python3 eight_readouts.py --out results/figures/eight_readouts.png
    python3 eight_readouts.py --redraw     # from the json beside the png
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_epistasis import gc_fraction, group_boot, out_of_fold_linear
from evo_probe import elements, kmers, out_of_fold
from single_variant import single_variants

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
COLOR = {"baseline": "#8a8a85", "likelihood": "#eb6834", "probe": "#2a78d6"}
KIND_LABEL = {"likelihood": "model score, zero-shot (raw Spearman)",
              "baseline": "sequence baseline (no model)",
              "probe": "model embedding probe, supervised"}

EVO_PRED = "results/evo2_7b_base/predictions.csv"
NTV3_PRED = "results/ntv3_650m_pre/predictions.csv"
EVO_EMB = "results/evo_probe"                 # blocks.26.mlp.l3, mean pooled
NTV3_DECONV = "results/ntv3_650m_deconv"      # deconv_7, one vector per base
NTV3_BOTTLENECK = "results/ntv3_650m_final"   # transformer block 11
NTV3_VARIANTS = "results/ntv3_650m_variants"  # block 11, alternate sequences
DB2 = "results/vp_db2_L11"                    # DNABERT-2 block 11 of 12
DB2_PRED = "results/dnabert2_117m/predictions.csv"  # dnabert2_score.py, then evaluate
EVO_VARIANT_PROBE = "results/vp_evo2/probe.txt"


def sha(seq):
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def interval(prediction, y, groups, n_boot, seed):
    rho = float(spearmanr(prediction, y).statistic)
    low, high = group_boot(groups, lambda i: spearmanr(prediction[i], y[i]).statistic,
                           n_boot, seed)
    return rho, [float(low), float(high)]


def row(label, kind, rho, ci, features, **extra):
    r = {"label": label, "kind": kind, "spearman": rho, "ci": ci, "features": features}
    r.update(extra)
    tail = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "(no interval)"
    print(f"  {label:44} {rho:+.4f}  {tail}", flush=True)
    return r


def gap(label, kind, why):
    print(f"  {label:44} not computed: {why}")
    return {"label": label, "kind": kind, "spearman": None, "ci": None,
            "not_computed": why}


def load_embedding(directory, ids):
    table = pd.read_csv(Path(directory) / "elements.csv")
    matrix = np.load(Path(directory) / "X_mean.npy")
    if len(table) != len(matrix):
        raise SystemExit(f"{directory}: {len(table)} rows but {len(matrix)} vectors")
    position = {s: i for i, s in enumerate(table.sequence_id)}
    missing = [s for s in ids if s not in position]
    if missing:
        raise SystemExit(f"{directory}: {len(missing)} elements have no embedding")
    return matrix[[position[s] for s in ids]]


def db2_reference_embedding(ids):
    """DNABERT-2 embedded the reference of every variant observation; the same
    reference appears once per variant with an identical vector, so one copy
    per element is taken and checked."""
    seqs = pd.read_csv(Path(DB2) / "sequences.csv")
    matrix = np.load(Path(DB2) / "X_wt_mean.npy")
    if len(seqs) != len(matrix):
        raise SystemExit(f"{DB2}: {len(seqs)} rows but {len(matrix)} vectors")
    first = {}
    for i, s in enumerate(seqs.wt_id):
        j = first.setdefault(s, i)
        if j != i and not np.array_equal(matrix[i], matrix[j]):
            raise SystemExit(f"{DB2}: reference {s[:12]} has two different vectors")
    missing = [s for s in ids if s not in first]
    if missing:
        raise SystemExit(f"{DB2}: {len(missing)} elements have no reference embedding")
    return matrix[[first[s] for s in ids]]


# ------------------------------------------------------------------ element

def element_panel(folds, n_boot, seed):
    evo = elements(EVO_PRED)
    ntv3 = elements(NTV3_PRED)
    if not evo.sequence_id.equals(ntv3.sequence_id):
        raise SystemExit("the two element tables are not aligned")
    y, groups = evo.activity.to_numpy(), evo.group.to_numpy()
    seqs, ids = evo.seq.tolist(), evo.sequence_id.tolist()
    print(f"element activity, n = {len(y)}")

    def fit(features):
        return out_of_fold(features, y, groups, folds)

    rows = []
    for label, score in (("Evo 2 log-likelihood", evo.s_wt.to_numpy()),
                         ("NTv3 650M pseudo-log-likelihood", ntv3.s_wt.to_numpy())):
        rho, ci = interval(score, y, groups, n_boot, seed)
        rows.append(row(label, "likelihood", rho, ci, 1, zero_shot=True))
    if Path(DB2_PRED).exists():
        db2 = elements(DB2_PRED)
        if not db2.sequence_id.equals(evo.sequence_id):
            raise SystemExit(f"{DB2_PRED}: element table is not aligned with Evo 2's")
        rho, ci = interval(db2.s_wt.to_numpy(), y, groups, n_boot, seed)
        rows.append(row("DNABERT-2 pseudo-log-likelihood", "likelihood", rho, ci, 1,
                        zero_shot=True, source=DB2_PRED))
    else:
        rows.append(gap("DNABERT-2 pseudo-log-likelihood", "likelihood",
                        "no DNABERT-2 scoring run exists (dnabert2_score.py)"))
    for label, features in (("GC content", gc_fraction(seqs).reshape(-1, 1)),
                            ("1/2/3-mer counts", kmers(seqs))):
        rho, ci = interval(fit(features), y, groups, n_boot, seed)
        rows.append(row(label, "baseline", rho, ci, int(features.shape[1])))
    probes = (
        ("DNABERT-2 probe (block 11 of 12)", db2_reference_embedding(ids), DB2),
        ("NTv3 650M probe (deconv_7)", load_embedding(NTV3_DECONV, ids), NTV3_DECONV),
        ("Evo 2 probe (blocks.26.mlp.l3)", load_embedding(EVO_EMB, ids), EVO_EMB),
    )
    for label, features, source in probes:
        rho, ci = interval(fit(features), y, groups, n_boot, seed)
        rows.append(row(label, "probe", rho, ci, int(features.shape[1]), source=source))
    return {"n": int(len(y)), "target": "measured log2 activity of the reference 200-mer",
            "protocol": "grouped five-fold, folds unshuffled, RidgeCV inside each "
                        "training fold (element_spearman.py); zero-shot rows are the "
                        "raw Spearman of the score; 95% interval resamples region groups",
            "readouts": rows}


# ------------------------------------------------------------------ variant

def variant_diff(directory, references, table):
    """h(alt) - h(ref) from an embed_variants.py set plus an element set, exactly
    as variant_probe_fit.py builds it."""
    Xv = np.load(Path(directory) / "X_mean.npy")
    sv = pd.read_csv(Path(directory) / "sequences.csv")
    Xr = np.load(Path(references) / "X_mean.npy")
    sr = pd.read_csv(Path(references) / "elements.csv")
    if len(sv) != len(Xv) or len(sr) != len(Xr) or Xv.shape[1] != Xr.shape[1]:
        raise SystemExit(f"{directory} / {references}: shapes do not agree")
    vi = {s: i for i, s in enumerate(sv.sequence_id)}
    ri = {s: i for i, s in enumerate(sr.sequence_id)}
    have = [(v in vi) and (r in ri) for v, r in zip(table.sequence_id, table.ref_id)]
    if not all(have):
        raise SystemExit(f"{directory}: {len(have) - sum(have)} variants lack a vector")
    return np.stack([Xv[vi[v]] - Xr[ri[r]] for v, r in zip(table.sequence_id, table.ref_id)])


def db2_variant_diff(table):
    """DNABERT-2 stored the difference directly, one row per (reference, variant)."""
    obs = pd.read_csv(Path(DB2) / "observations.csv")
    Xd = np.load(Path(DB2) / "X_d_mean.npy")
    if len(obs) != len(Xd):
        raise SystemExit(f"{DB2}: {len(obs)} rows but {len(Xd)} vectors")
    where = {(w, m): i for i, (w, m) in enumerate(zip(obs.wt_id, obs.mut_id))}
    keys = list(zip(table.ref_id, table.sequence_id))
    missing = [k for k in keys if k not in where]
    if missing:
        raise SystemExit(f"{DB2}: {len(missing)} variants have no difference vector")
    return Xd[[where[k] for k in keys]]


def carried_evo_variant_probe():
    text = Path(EVO_VARIANT_PROBE).read_text()
    match = re.search(r"^difference, mean pooled\s+([+-][\d.]+)\s", text, re.M)
    n = re.search(r"^n=(\d+)", text, re.M)
    if not match or not n:
        return None
    return float(match.group(1)), int(n.group(1))


def variant_panel(folds, n_boot, seed):
    evo = single_variants(EVO_PRED)
    ntv3 = single_variants(NTV3_PRED)
    if not evo.sequence_id.equals(ntv3.sequence_id):
        raise SystemExit("the two single-variant tables are not aligned")
    table = evo.assign(ref_id=[sha(s) for s in evo.seq_ref])
    y, groups = table.y.to_numpy(), table.group_id.to_numpy()
    seqs, refs = table.seq.tolist(), table.seq_ref.tolist()
    print(f"single-variant effect, n = {len(y)}")

    def fit(features):
        return out_of_fold_linear(features, y, groups, folds,
                                  ridge=features.shape[1] > 1, seed=seed)

    rows = []
    for label, score in (("Evo 2 delta log-likelihood", evo.delta_score.to_numpy()),
                         ("NTv3 650M delta pseudo-log-likelihood", ntv3.delta_score.to_numpy())):
        rho, ci = interval(score, y, groups, n_boot, seed)
        rows.append(row(label, "likelihood", rho, ci, 1, zero_shot=True))
    if Path(DB2_PRED).exists():
        db2 = single_variants(DB2_PRED)
        if not db2.sequence_id.equals(evo.sequence_id):
            raise SystemExit(f"{DB2_PRED}: single-variant table is not aligned with Evo 2's")
        rho, ci = interval(db2.delta_score.to_numpy(), y, groups, n_boot, seed)
        rows.append(row("DNABERT-2 delta pseudo-log-likelihood", "likelihood", rho, ci, 1,
                        zero_shot=True, source=DB2_PRED))
    else:
        rows.append(gap("DNABERT-2 delta pseudo-log-likelihood", "likelihood",
                        "no DNABERT-2 scoring run exists (dnabert2_score.py)"))
    for label, features in (("GC content", gc_fraction(seqs).reshape(-1, 1)),
                            ("1/2/3-mer delta", kmers(seqs) - kmers(refs))):
        rho, ci = interval(fit(features), y, groups, n_boot, seed)
        rows.append(row(label, "baseline", rho, ci, int(features.shape[1])))

    db2 = db2_variant_diff(table)
    rho, ci = interval(fit(db2), y, groups, n_boot, seed)
    rows.append(row("DNABERT-2 probe (block 11 of 12)", "probe", rho, ci,
                    int(db2.shape[1]), source=DB2))
    bottleneck = variant_diff(NTV3_VARIANTS, NTV3_BOTTLENECK, table)
    rho, ci = interval(fit(bottleneck), y, groups, n_boot, seed)
    rows.append(row("NTv3 650M probe (block 11, bottleneck)", "probe", rho, ci,
                    int(bottleneck.shape[1]), source=NTV3_VARIANTS,
                    substitute="deconv-layer variant embeddings do not exist; this is "
                               "the transformer bottleneck, block 11"))
    carried = carried_evo_variant_probe()
    if carried is None:
        rows.append(gap("Evo 2 probe (blocks.26.mlp.l3)", "probe",
                        "no variant embeddings"))
    else:
        rows.append(row("Evo 2 probe (blocks.26.mlp.l3)", "probe", carried[0], None, None,
                        n=carried[1], other_protocol=True, source=EVO_VARIANT_PROBE))
    return {"n": int(len(y)), "target": "measured log2 effect of one substitution",
            "protocol": "grouped five-fold shuffled seed 0; ridge for multi-column "
                        "features (single_variant_spearman.py); zero-shot rows are the "
                        "raw Spearman of the delta score; 95% interval resamples region "
                        "groups; probe feature is h(alt) - h(ref), mean pooled",
            "readouts": rows}


# ------------------------------------------------------------------ drawing

TICK = {
    "Evo 2 log-likelihood": "Evo 2\nzero-shot",
    "NTv3 650M pseudo-log-likelihood": "NTv3 650M\nzero-shot",
    "DNABERT-2 pseudo-log-likelihood": "DNABERT-2\nzero-shot",
    "Evo 2 delta log-likelihood": "Evo 2\nzero-shot",
    "NTv3 650M delta pseudo-log-likelihood": "NTv3 650M\nzero-shot",
    "DNABERT-2 delta pseudo-log-likelihood": "DNABERT-2\nzero-shot",
    "GC content": "GC\ncontent",
    "1/2/3-mer counts": "1/2/3-mer\ncounts",
    "1/2/3-mer delta": "1/2/3-mer\ndelta",
    "DNABERT-2 probe (block 11 of 12)": "DNABERT-2\nprobe\nblock 11 (last)",
    "NTv3 650M probe (deconv_7)": "NTv3 650M\nprobe\ndeconv_7 (last)",
    "NTv3 650M probe (block 11, bottleneck)": "NTv3 650M\nprobe\nblock 11 (bottleneck)†",
    "Evo 2 probe (blocks.26.mlp.l3)": "Evo 2\nprobe\nblocks.26.mlp.l3",
}


def draw_panel(axis, data, title, ylabel):
    rows = data["readouts"]
    x = np.arange(len(rows))
    done = [(i, r) for i, r in enumerate(rows) if r["spearman"] is not None]
    gaps = [(i, r) for i, r in enumerate(rows) if r["spearman"] is None]
    withci = [(i, r) for i, r in done if r.get("ci")]
    noci = [(i, r) for i, r in done if not r.get("ci")]

    bars = axis.bar([i for i, _ in done], [r["spearman"] for _, r in done],
                    color=[COLOR[r["kind"]] for _, r in done], width=0.62, zorder=3)
    axis.errorbar([i for i, _ in withci], [r["spearman"] for _, r in withci],
                  yerr=[[r["spearman"] - r["ci"][0] for _, r in withci],
                        [r["ci"][1] - r["spearman"] for _, r in withci]],
                  fmt="none", ecolor=INK, capsize=4, lw=1.1, zorder=4)
    spans = [r["ci"][1] for _, r in withci] + [r["spearman"] for _, r in noci]
    lows = [r["ci"][0] for _, r in withci] + [r["spearman"] for _, r in noci]
    top = max(spans) + 0.09 * (max(spans) - min(0, min(lows)))
    bottom = min(0, min(lows)) - 0.13 * (max(spans) - min(0, min(lows)))
    pad = 0.012 * (top - bottom)
    for i, r in withci:
        # Above the upper cap for a positive bar, below the lower cap for a
        # negative one, so the number never sits inside the bar.
        if r["spearman"] >= 0:
            axis.text(i, r["ci"][1] + pad, f"{r['spearman']:+.3f}",
                      ha="center", va="bottom", fontsize=9.5, color=INK)
        else:
            axis.text(i, r["ci"][0] - pad, f"{r['spearman']:+.3f}",
                      ha="center", va="top", fontsize=9.5, color=INK)
    for i, r in noci:
        bar = bars[[j for j, _ in done].index(i)]
        bar.set_hatch("///")
        bar.set_edgecolor("white")
        axis.text(i, r["spearman"] + pad, f"{r['spearman']:+.3f}*",
                  ha="center", fontsize=9.5, color=INK)
    for i, r in gaps:
        axis.add_patch(plt.Rectangle((i - 0.31, 0), 0.62, top * 0.42, facecolor="none",
                                     edgecolor=GRID, lw=1, ls=(0, (3, 3)), zorder=3))
        axis.text(i, top * 0.21, "no scoring\nrun", ha="center", va="center",
                  fontsize=8.5, color=MUTED, style="italic")
    axis.axhline(0, color=MUTED, lw=1, zorder=2)
    axis.set_xticks(x, [TICK.get(r["label"], r["label"]) for r in rows],
                    fontsize=8.8, color=INK)
    for tick, r in zip(axis.get_xticklabels(), rows):
        tick.set_color(MUTED if r["spearman"] is None else INK)
    axis.set_ylabel(ylabel, fontsize=10, color=INK)
    axis.set_ylim(bottom, top)
    axis.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, length=3)
    axis.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10)


def draw(element, variant, path):
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11.5, 11.8))
    figure.patch.set_facecolor("white")
    draw_panel(top, element,
               f"A   Predicting element activity from the 200-mer  "
               f"(n = {element['n']:,} elements)\n"
               "      out of fold, grouped five-fold unshuffled, 95% interval over region groups",
               "Spearman with measured element activity")
    draw_panel(bottom, variant,
               f"B   Predicting the effect of a single substitution  "
               f"(n = {variant['n']:,} variants)\n"
               "      out of fold, grouped five-fold shuffled seed 0, 95% interval over "
               "region groups; probe feature is h(alt) − h(ref)",
               "Spearman with the measured single-variant effect")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLOR[k], label=v)
               for k, v in KIND_LABEL.items()]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=9, color="white",
                              mec=GRID, label="not computed"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                  fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.004))
    notes = [
        "Zero-shot bars are the raw Spearman of the model's own score. Baselines and "
        "probes are fitted out of fold; probes are mean-pooled embeddings at the layer named.",
        "† No NTv3 650M variant embeddings exist at the deconv layer; panel B shows "
        "the transformer bottleneck (block 11) instead.\n   In panel A the same checkpoint "
        "reads +0.440 at block 11 and +0.485 at deconv_7, so the panel B bar is likely to move.",
        "* Evo 2's variant probe predates panel B's protocol and its embeddings were never "
        "committed: n = 5,428, its own folds, no interval. Read it beside the others, not "
        "against them.",
    ]
    figure.text(0.012, 0.048, "\n".join(notes), fontsize=8.2, color=MUTED,
                ha="left", va="bottom", linespacing=1.5)
    figure.tight_layout(rect=(0, 0.12, 1, 1), h_pad=3.5)
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/figures/eight_readouts.png", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redraw", action="store_true",
                        help="draw from the json beside --out instead of refitting")
    args = parser.parse_args()
    store = args.out.with_suffix(".json")
    if args.redraw:
        data = json.loads(store.read_text())
    else:
        data = {"element": element_panel(args.folds, args.bootstrap, args.seed),
                "variant": variant_panel(args.folds, args.bootstrap, args.seed)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(data, indent=2) + "\n")
    draw(data["element"], data["variant"], args.out)


if __name__ == "__main__":
    main()
