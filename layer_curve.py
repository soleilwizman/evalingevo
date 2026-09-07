#!/usr/bin/env python3
"""Every layer, and the one control that says what a probe win means.

Two questions, both answered on CPU from matrices already on disk.

  1. Which layer? Prints every layer's out-of-fold Spearman, so no layer has to
     be guessed. Layer 0 is pre-transformer: whatever it scores is what the
     input representation alone can do.

  2. What does the embedding know that letter-counting doesn't? Regress activity
     on 1-2-3 k-mer counts first, then predict the leftover with the embedding.
     A probe that only beats k-mers by having more features will score near zero
     here. A probe carrying non-compositional information will not.

    python3 layer_curve.py results/ntv3_650m_sweep
    python3 layer_curve.py results/evo_probe          # single-layer dirs work too
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_probe import elements, kmers, out_of_fold, paired_interval


def layer_matrices(directory, pooling):
    swept = sorted(directory.glob(f"X_{pooling}_L*.npy"),
                   key=lambda f: int(f.stem.split("_L")[1]))
    if swept:
        return {int(p.stem.split("_L")[1]): p for p in swept}
    single = directory / f"X_{pooling}.npy"
    if single.exists():
        return {None: single}
    raise SystemExit(f"no X_{pooling}*.npy in {directory}")


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ntv3_650m_sweep")
    pooling = sys.argv[2] if len(sys.argv) > 2 else "mean"
    matrices = layer_matrices(directory, pooling)

    el = pd.read_csv(directory / "elements.csv")
    full = elements().set_index("sequence_id")
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values

    km = kmers(seqs)
    km_pred = out_of_fold(km, y, g, 5)
    base = spearmanr(km_pred, y).statistic
    # What k-mers cannot explain. Out-of-fold, so the residual is honest.
    residual = y - km_pred

    print(f"{directory}   pooling {pooling}   n={len(el)}   {len(matrices)} layer(s)")
    print(f"1-2-3 k-mer counts vs activity: {base:+.4f}\n")
    print("layer   bad rows   vs activity   vs k-mer residual")

    rows = []
    for index, path in matrices.items():
        X = np.load(path)
        finite = np.isfinite(X)
        bad = int((~finite.all(1)).sum())
        if bad == len(X):
            print(f"{str(index):>5}   {bad:>8}   every row non-finite, skipped")
            continue
        keep = finite.all(1)
        direct = spearmanr(out_of_fold(X[keep], y[keep], g[keep], 5), y[keep]).statistic
        extra = spearmanr(out_of_fold(X[keep], residual[keep], g[keep], 5),
                          residual[keep]).statistic
        rows.append({"layer": index, "direct": float(direct), "residual": float(extra),
                     "n": int(keep.sum())})
        suffix = "" if bad == 0 else f"   (on {keep.sum()} rows)"
        print(f"{str(index):>5}   {bad:>8}   {direct:+.4f}        {extra:+.4f}{suffix}",
              flush=True)

    if not rows:
        raise SystemExit("no usable layers")

    best_direct = max(rows, key=lambda r: r["direct"])
    best_extra = max(rows, key=lambda r: r["residual"])
    print(f"\nbest against activity:      layer {best_direct['layer']} "
          f"at {best_direct['direct']:+.4f}  ({best_direct['direct'] - base:+.4f} vs k-mers)")
    print(f"best against k-mer residual: layer {best_extra['layer']} "
          f"at {best_extra['residual']:+.4f}")

    X = np.load(matrices[best_direct["layer"]])
    keep = np.isfinite(X).all(1)
    low, high = paired_interval(out_of_fold(X[keep], y[keep], g[keep], 5),
                                km_pred[keep], y[keep], g[keep])
    print(f"\nbest layer minus k-mers: {best_direct['direct'] - base:+.4f}  "
          f"95% interval [{low:+.4f}, {high:+.4f}]")
    print("Both numbers are optimistic: the layer was chosen after seeing them.\n")

    if 0 in [r["layer"] for r in rows]:
        zero = next(r for r in rows if r["layer"] == 0)
        deeper = [r for r in rows if r["layer"] not in (0, None)]
        if deeper and max(r["direct"] for r in deeper) <= zero["direct"]:
            print("No transformer block beats layer 0. On this task the blocks add "
                  "nothing over the input representation.")
        elif deeper:
            top = max(deeper, key=lambda r: r["direct"])
            print(f"Layer {top['layer']} beats layer 0 by "
                  f"{top['direct'] - zero['direct']:+.4f}. That gap is what the "
                  "transformer blocks contribute.")


if __name__ == "__main__":
    main()
