#!/usr/bin/env python3
"""CPU: probe variant embeddings against the measured single-variant effect.

Consumes an embed_variants.py directory. For each of the 5,666 single variants
it forms the difference vector h(alt) - h(ref) from the two sequences of that
same pair, which cancels the static genomic background: an element-level probe
predicts reference activity at about +0.5, so a probe fed raw embeddings would
mostly learn which region it was looking at rather than what the variant did.

Reported against the same baselines and the same folds as single_variant.py, so
the number lands on the existing figures without a protocol change.

    python3 variant_probe.py --embeddings results/evo_variants \
        --label "Evo 2 probe" --out results/evo_variants/variant_probe.json
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


def difference_vectors(embeddings, table, pooling):
    """h(alt) - h(ref), one row per single variant, from one checkpoint."""
    directory = Path(embeddings)
    matrix = np.load(directory / f"X_{pooling}.npy", mmap_mode="r")
    index = pd.read_csv(directory / "sequences.csv")
    progress = directory / "progress.json"
    if progress.exists():
        done = json.loads(progress.read_text())["done"]
        if done < len(index):
            raise SystemExit(f"{embeddings} is incomplete: {done} of {len(index)} rows "
                             "embedded. Re-run embed_variants.py to finish it.")
    if len(index) != len(matrix):
        raise SystemExit(f"{embeddings}: {len(index)} rows but {len(matrix)} embeddings")
    position = {sequence: row for row, sequence in enumerate(index.sequence_id)}
    missing = {s for s in table.sequence_id if s not in position}
    missing |= {s for s in table.wt_id if s not in position} if "wt_id" in table else set()
    if missing:
        raise SystemExit(f"{len(missing)} sequences have no embedding. The directory "
                         "was probably built without the variant sequences.")
    alt = np.asarray(matrix[[position[s] for s in table.sequence_id]], dtype=np.float64)
    ref = np.asarray(matrix[[position[s] for s in table.reference_id]], dtype=np.float64)
    return alt - ref


def probe(embeddings, label, pooling="mean", predictions="results/evo2_7b_base/predictions.csv",
          folds=5, seed=0, n_boot=1000):
    table = single_variants(predictions)
    # single_variants keeps the reference sequence but not its hash; recover it.
    quartets = pd.read_csv("data/quartets.csv.gz", usecols=["seq_wt", "id_wt"])
    reference = dict(zip(quartets.seq_wt, quartets.id_wt))
    table = table.assign(reference_id=table.seq_ref.map(reference))
    if table.reference_id.isna().any():
        raise SystemExit("a reference sequence has no id in quartets.csv.gz")

    y, groups = table.y.to_numpy(), table.group_id.to_numpy()
    delta = difference_vectors(embeddings, table, pooling)
    baseline = kmers(table.seq.tolist()) - kmers(table.seq_ref.tolist())
    meta = json.loads((Path(embeddings) / "meta.json").read_text())

    def fit(x):
        return out_of_fold_linear(x, y, groups, folds, ridge=x.shape[1] > 1, seed=seed)

    fits = {
        f"{label} difference vector": fit(delta),
        "delta k-mers": fit(baseline),
        f"{label} plus delta k-mers": fit(np.hstack([delta, baseline])),
    }
    rows = {}
    for name, prediction in fits.items():
        rho = float(spearmanr(prediction, y).statistic)
        low, high = group_boot(groups, lambda i: spearmanr(prediction[i], y[i]).statistic,
                               n_boot, seed)
        rows[name] = {"spearman": rho, "ci": [float(low), float(high)],
                      "rmse": float(np.sqrt(np.mean((y - prediction) ** 2)))}
        print(f"  {name:44} {rho:+.4f}  [{low:+.4f}, {high:+.4f}]", flush=True)

    margins = {}
    for name in (f"{label} difference vector", f"{label} plus delta k-mers"):
        low, high = paired_interval(fits[name], fits["delta k-mers"], y, groups)
        margin = rows[name]["spearman"] - rows["delta k-mers"]["spearman"]
        margins[name] = {"margin": margin, "interval": [float(low), float(high)],
                         "verdict": "beats delta k-mers" if low > 0 else
                                    "worse than delta k-mers" if high < 0 else
                                    "not distinguishable from delta k-mers"}
        print(f"  {name} minus delta k-mers: {margin:+.4f} "
              f"[{low:+.4f}, {high:+.4f}] -> {margins[name]['verdict']}", flush=True)

    return {"label": label, "embeddings": str(embeddings), "pooling": pooling,
            "checkpoint": meta.get("checkpoint"), "layer": meta.get("layer"),
            "width": int(delta.shape[1]), "n": int(len(table)),
            "target": "measured log2 effect of one substitution, ref/alt coding",
            "feature": "h(alt) - h(ref), pooled, same checkpoint for both",
            "protocol": "grouped five-fold shuffled seed 0; ridge for multi-column "
                        "features; 95% interval resamples whole region groups",
            "readouts": rows, "margin_over_kmer_delta": margins}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pooling", default="mean", choices=("mean", "last"))
    parser.add_argument("--predictions", default="results/evo2_7b_base/predictions.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = probe(args.embeddings, args.label, args.pooling, args.predictions,
                   args.folds, args.seed, args.bootstrap)
    out = Path(args.out or Path(args.embeddings) / "variant_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
