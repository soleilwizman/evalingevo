#!/usr/bin/env python3
"""CPU: probe variant embeddings against the measured single-variant effect.

Takes the variant embeddings from embed_variants.py and the reference
embeddings already committed, and forms h(alt) - h(ref) per variant. The
difference cancels the static genomic background: an element-level probe
predicts reference activity at about +0.5, so a probe fed raw embeddings would
mostly learn which region it was looking at rather than what the variant did.

The two directories must name the same checkpoint, layer and revision. That is
checked, not assumed; subtracting embeddings from two different runs would
produce a number that means nothing.

Reported against the same baselines and the same folds as single_variant.py, so
the number lands on the existing figures without a protocol change.

    python3 variant_probe.py --embeddings results/evo_variants \
        --reference results/evo_probe --label "Evo 2 probe"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_epistasis import group_boot, out_of_fold_linear
from evo_probe import kmers, paired_interval
from single_variant import single_variants


def load_matrix(directory, pooling):
    """Embedding matrix plus a sequence_id -> row index, from either layout."""
    directory = Path(directory)
    matrix = np.load(directory / f"X_{pooling}.npy", mmap_mode="r")
    listing = directory / "sequences.csv"
    if not listing.exists():
        listing = directory / "elements.csv"      # the committed reference layout
    if not listing.exists():
        raise SystemExit(f"{directory}: no sequences.csv or elements.csv")
    index = pd.read_csv(listing)
    progress = directory / "progress.json"
    if progress.exists():
        done = json.loads(progress.read_text())["done"]
        if done < len(index):
            raise SystemExit(f"{directory} is incomplete: {done} of {len(index)} rows "
                             "embedded. Re-run embed_variants.py to finish it.")
    if len(index) != len(matrix):
        raise SystemExit(f"{directory}: {len(index)} rows but {len(matrix)} embeddings")
    meta = json.loads((directory / "meta.json").read_text())
    return matrix, {s: i for i, s in enumerate(index.sequence_id)}, meta


def describe(meta):
    """Checkpoint and layer as one comparable tuple, across both meta layouts."""
    return (meta.get("checkpoint"),
            meta.get("representation") or meta.get("layer"),
            meta.get("revision"))


def difference_vectors(embeddings, reference, table, pooling):
    """h(alt) - h(ref), one row per single variant, from one checkpoint."""
    alt_matrix, alt_index, alt_meta = load_matrix(embeddings, pooling)
    if reference is None:
        ref_matrix, ref_index, ref_meta = alt_matrix, alt_index, alt_meta
    else:
        ref_matrix, ref_index, ref_meta = load_matrix(reference, pooling)
        if describe(alt_meta) != describe(ref_meta):
            raise SystemExit(
                "the variant and reference embeddings are not from the same run:\n"
                f"  {embeddings}: {describe(alt_meta)}\n"
                f"  {reference}: {describe(ref_meta)}\n"
                "Subtracting these would be meaningless. Re-embed with matching "
                "--checkpoint and --layer, or pass --include-reference.")
        if alt_matrix.shape[1] != ref_matrix.shape[1]:
            raise SystemExit(f"width mismatch: {alt_matrix.shape[1]} vs {ref_matrix.shape[1]}")
    missing_alt = [s for s in table.sequence_id if s not in alt_index]
    missing_ref = [s for s in table.reference_id if s not in ref_index]
    if missing_alt:
        raise SystemExit(f"{len(missing_alt)} variant sequences have no embedding in "
                         f"{embeddings}")
    if missing_ref:
        raise SystemExit(f"{len(missing_ref)} reference sequences have no embedding in "
                         f"{reference or embeddings}")
    alt = np.asarray(alt_matrix[[alt_index[s] for s in table.sequence_id]], dtype=np.float64)
    ref = np.asarray(ref_matrix[[ref_index[s] for s in table.reference_id]], dtype=np.float64)
    return alt, ref, alt_meta


def probe(embeddings, label, reference=None, pooling="mean",
          predictions="results/evo2_7b_base/predictions.csv",
          folds=5, seed=0, n_boot=1000, include_alternate=False):
    table = single_variants(predictions)
    # single_variants keeps the reference sequence but not its hash; recover it.
    quartets = pd.read_csv("data/quartets.csv.gz", usecols=["seq_wt", "id_wt"])
    reference_ids = dict(zip(quartets.seq_wt, quartets.id_wt))
    table = table.assign(reference_id=table.seq_ref.map(reference_ids))
    if table.reference_id.isna().any():
        raise SystemExit("a reference sequence has no id in quartets.csv.gz")

    y, groups = table.y.to_numpy(), table.group_id.to_numpy()
    alt, ref, meta = difference_vectors(embeddings, reference, table, pooling)
    delta = alt - ref
    baseline = kmers(table.seq.tolist()) - kmers(table.seq_ref.tolist())

    def fit(x):
        return out_of_fold_linear(x, y, groups, folds, ridge=x.shape[1] > 1, seed=seed)

    # The reference-only row is the control that makes the difference vector
    # mean something. Each reference carries about two variants, so a probe on
    # h(ref) alone can only learn how mutable the element is, never which
    # substitution happened. If it matches the difference vector, the
    # subtraction bought nothing and the probe is reading the background.
    fits = {
        f"{label} difference vector": fit(delta),
        f"{label} reference only, control": fit(ref),
        "delta k-mers": fit(baseline),
        f"{label} plus delta k-mers": fit(np.hstack([delta, baseline])),
    }
    if include_alternate:
        fits[f"{label} alternate only, control"] = fit(alt)
    rows = {}
    for name, prediction in fits.items():
        rho = float(spearmanr(prediction, y).statistic)
        low, high = group_boot(groups, lambda i: spearmanr(prediction[i], y[i]).statistic,
                               n_boot, seed)
        rows[name] = {"spearman": rho, "ci": [float(low), float(high)],
                      "rmse": float(np.sqrt(np.mean((y - prediction) ** 2)))}
        print(f"  {name:44} {rho:+.4f}  [{low:+.4f}, {high:+.4f}]", flush=True)

    margins = {}
    control = rows[f"{label} reference only, control"]["spearman"]
    difference = rows[f"{label} difference vector"]["spearman"]
    low, high = paired_interval(fits[f"{label} difference vector"],
                                fits[f"{label} reference only, control"], y, groups)
    over_control = {"margin": difference - control, "interval": [float(low), float(high)],
                    "verdict": "the difference vector carries variant-specific signal "
                               "the reference alone does not" if low > 0 else
                               "no better than the reference alone: the probe is reading "
                               "the background, not the substitution" if high < 0 else
                               "not distinguishable from the reference alone"}
    print(f"  difference minus reference-only control: {over_control['margin']:+.4f} "
          f"[{low:+.4f}, {high:+.4f}] -> {over_control['verdict']}", flush=True)

    for name in (f"{label} difference vector", f"{label} plus delta k-mers"):
        low, high = paired_interval(fits[name], fits["delta k-mers"], y, groups)
        margin = rows[name]["spearman"] - rows["delta k-mers"]["spearman"]
        margins[name] = {"margin": margin, "interval": [float(low), float(high)],
                         "verdict": "beats delta k-mers" if low > 0 else
                                    "worse than delta k-mers" if high < 0 else
                                    "not distinguishable from delta k-mers"}
        print(f"  {name} minus delta k-mers: {margin:+.4f} "
              f"[{low:+.4f}, {high:+.4f}] -> {margins[name]['verdict']}", flush=True)

    return {"label": label, "embeddings": str(embeddings),
            "reference_embeddings": str(reference) if reference else str(embeddings),
            "pooling": pooling,
            "checkpoint": meta.get("checkpoint"), "layer": meta.get("layer"),
            "width": int(delta.shape[1]), "n": int(len(table)),
            "target": "measured log2 effect of one substitution, ref/alt coding",
            "feature": "h(alt) - h(ref), pooled, same checkpoint for both",
            "protocol": "grouped five-fold shuffled seed 0; ridge for multi-column "
                        "features; 95% interval resamples whole region groups",
            "readouts": rows, "margin_over_kmer_delta": margins,
            "margin_over_reference_only": over_control}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True,
                        help="an embed_variants.py output directory")
    parser.add_argument("--reference", default=None,
                        help="the committed reference embeddings for the SAME checkpoint "
                             "and layer, e.g. results/evo_probe. Omit only if the variant "
                             "run used --include-reference.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--pooling", default="mean", choices=("mean", "last"))
    parser.add_argument("--predictions", default="results/evo2_7b_base/predictions.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--include-alternate", action="store_true",
                        help="also fit h(alt) alone; another background control, and "
                             "another wide ridge fit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = probe(args.embeddings, args.label, args.reference, args.pooling,
                   args.predictions, args.folds, args.seed, args.bootstrap,
                   args.include_alternate)
    out = Path(args.out or Path(args.embeddings) / "variant_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
