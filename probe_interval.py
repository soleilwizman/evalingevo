#!/usr/bin/env python3
"""Just the margin over word counts and its interval, for when the null is too slow.

`ntv3_probe.py probe` prints its permutation null before the paired interval, and that
null refits the full-width ridge 20 times over 5 folds: about 7 minutes on a 2595 x 1536
matrix, against 4 seconds for the interval itself. (`evo_probe.py` permutes the
84-feature k-mer matrix instead, which is why the Evo probe never felt slow. The
wide-matrix version is the better control, it measures the probe's own capacity to fit
noise, so this script skips it rather than replacing it.)

Same folds, same estimator, same bootstrap as the full probe, so the number is
comparable to every committed probe.txt.

    python3 probe_interval.py results/ntv3_650m_deconv
    python3 probe_interval.py results/evo_probe --pooling mean
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_probe import elements, kmers, out_of_fold, paired_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("embeddings")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    parser.add_argument("--seeds", type=int, default=1,
                        help="repeat the bootstrap under this many seeds. Near zero the "
                             "printed verdict can be decided by the seed rather than the "
                             "data, so use this before calling a narrow margin a win")
    args = parser.parse_args()

    directory = Path(args.embeddings)
    X = np.load(directory / f"X_{args.pooling}.npy")
    el = pd.read_csv(directory / "elements.csv")
    if len(X) != len(el):
        raise SystemExit(f"embeddings ({len(X)}) and elements ({len(el)}) disagree")
    full = elements().set_index("sequence_id")
    missing = set(el.sequence_id) - set(full.index)
    if missing:
        raise SystemExit(f"{len(missing)} embedded sequences are not in the element table")
    seqs = full.seq.loc[el.sequence_id].values
    y, groups = el.activity.values, el.group.values

    probe = out_of_fold(X, y, groups, args.folds)
    words = out_of_fold(kmers(seqs), y, groups, args.folds)
    rho_probe = float(spearmanr(probe, y).statistic)
    rho_words = float(spearmanr(words, y).statistic)
    bounds = [paired_interval(probe, words, y, groups, args.n_boot, seed)
              for seed in range(args.seeds)]
    low, high = bounds[0]

    print(f"{directory}   n={len(el)}   pooling {args.pooling}   width {X.shape[1]}")
    print(f"  probe        {rho_probe:+.4f}")
    print(f"  word counts  {rho_words:+.4f}")
    print(f"  margin       {rho_probe - rho_words:+.4f}  95% interval "
          f"[{low:+.4f}, {high:+.4f}]")
    if args.seeds > 1:
        lows = np.array([b[0] for b in bounds])
        cleared = int((lows > 0).sum())
        print(f"  lower bound over {args.seeds} seeds: min {lows.min():+.4f}, "
              f"max {lows.max():+.4f}, clears zero {cleared}/{args.seeds}")
        if cleared not in (0, args.seeds):
            raise SystemExit("  ON THE BOUNDARY: the verdict changes with the bootstrap "
                             "seed, so this is not a win. Report the margin and the "
                             "seed spread, not a pass/fail.")
    print("  clears zero: the representation beats letter counting."
          if low > 0 else
          "  crosses zero: not distinguishable from letter counting on this evidence.")


if __name__ == "__main__":
    main()
