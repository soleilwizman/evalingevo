#!/usr/bin/env python3
"""Every layer, and the one control that says what a probe win means.

Two questions, both answered on CPU from matrices already on disk.

  1. Which layer? Prints every layer's out-of-fold Spearman, so no layer has to
     be guessed. Interpret layer indices using the run's meta.json, not a
     different sweep's numbering.

  2. What does the embedding know that letter-counting doesn't? Regress activity
     on 1-2-3 k-mer counts first, then predict the leftover with the embedding.
     A probe that only beats k-mers by having more features will score near zero
     here. A probe carrying non-compositional information will not.

    python3 scripts/layer_curve.py results/ntv3_650m_sweep
    python3 scripts/layer_curve.py results/evo_probe          # single-layer dirs work too
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import load_matrix
from element_data import elements, validate_elements
from evo_probe import kmers, paired_interval
from scipy.stats import spearmanr
from validation import out_of_fold, residual_predictions, selected_predictions


def layer_matrices(directory, pooling):
    swept = sorted(directory.glob(f"X_{pooling}_L*.npy"), key=lambda f: int(f.stem.split("_L")[1]))
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
    validate_elements(el, full)
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values

    km = kmers(seqs)
    km_pred = out_of_fold(km, y, g, 5)
    base = spearmanr(km_pred, y).statistic

    print(f"{directory}   pooling {pooling}   n={len(el)}   {len(matrices)} layer(s)")
    print(f"1-2-3 k-mer counts vs activity: {base:+.4f}\n")
    print("layer   bad rows   vs activity   vs k-mer residual")

    rows = []
    for index, path in matrices.items():
        X = load_matrix(path, el, ["sequence_id"])
        direct = spearmanr(out_of_fold(X, y, g, 5), y).statistic
        residual_pred, residual = residual_predictions(X, km, y, g, 5)
        extra = spearmanr(residual_pred, residual).statistic
        rows.append(
            {
                "layer": index,
                "direct": float(direct),
                "residual": float(extra),
                "n": len(X),
            }
        )
        print(
            f"{str(index):>5}   {0:>8}   {direct:+.4f}        {extra:+.4f}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no usable layers")

    best_direct = max(rows, key=lambda r: r["direct"])
    best_extra = max(rows, key=lambda r: r["residual"])
    print(
        f"\nbest against activity:      layer {best_direct['layer']} "
        f"at {best_direct['direct']:+.4f}  ({best_direct['direct'] - base:+.4f} vs k-mers)"
    )
    print(
        f"best against k-mer residual: layer {best_extra['layer']} at {best_extra['residual']:+.4f}"
    )

    selected, choices = selected_predictions(
        {str(index): np.load(path) for index, path in matrices.items()}, y, g, 5
    )
    low, high = paired_interval(selected, km_pred, y, g)
    margin = spearmanr(selected, y).statistic - base
    print(
        f"\ninner-selected layer minus k-mers: {margin:+.4f}, "
        f"95% interval [{low:+.4f}, {high:+.4f}], choices {choices}"
    )
    print("Individual layer rankings are exploratory; the selected comparison uses nested groups.")

    if 0 in [r["layer"] for r in rows]:
        zero = next(r for r in rows if r["layer"] == 0)
        deeper = [r for r in rows if r["layer"] not in (0, None)]
        if deeper and max(r["direct"] for r in deeper) <= zero["direct"]:
            print(
                "No tested deeper stage beats layer 0. On this task these stages add "
                "nothing over the input representation."
            )
        elif deeper:
            top = max(deeper, key=lambda r: r["direct"])
            print(
                f"Layer {top['layer']} beats layer 0 by "
                f"{top['direct'] - zero['direct']:+.4f}. That gap is what the "
                "tested deeper stage contributes."
            )


if __name__ == "__main__":
    main()
