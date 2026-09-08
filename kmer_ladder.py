#!/usr/bin/env python3
"""Does any embedding beat letter counting once the letter counting is done properly?

Every probe in this repo is scored against `kmers` in evo_probe.py: overlapping 1, 2 and
3-mer counts, 84 features. That is a weak baseline for a 200-mer. This script raises it
and re-runs the same paired interval, so a probe's margin is measured against a baseline
that had a fair chance.

The answer changes the headline. A plain 6-mer count vector reaches +0.5294, above Evo 2's
+0.5051 probe, so Evo's +0.0490 margin over 84 features is an artifact of the baseline
rather than evidence its representations carry non-compositional information. Only NTv3's
early conv tower survives every rung.

Same folds (GroupKFold, no shuffle), same RidgeCV inside each training fold, same group
bootstrap, and the 10-seed robustness check from evo_probe.report_margin.

    python3 kmer_ladder.py
    python3 kmer_ladder.py --max-k 4        # stop at 1-4-mers, about 3 minutes
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_probe import elements, out_of_fold, paired_interval, VERDICT_SEEDS

REPRESENTATIONS = (
    ("NTv3 650M conv_2", "results/ntv3_650m_sweep/X_mean_L1.npy"),
    ("NTv3 650M deconv_7", "results/ntv3_650m_deconv/X_mean.npy"),
    ("NTv3 650M transformer_11", "results/ntv3_650m_final/X_mean.npy"),
    ("NTv3 100M transformer_5", "results/ntv3_100m_final/X_mean.npy"),
    ("Evo 2 blocks.26.mlp.l3", "results/evo_probe/X_mean.npy"),
)
ELEMENTS = "results/ntv3_650m_sweep/elements.csv"


def kmer_matrix(sequences, ks):
    """Overlapping counts for every k in ks. 4**k columns per k, in fixed order."""
    vocab = ["".join(c) for k in ks for c in itertools.product("ACGT", repeat=k)]
    index = {word: i for i, word in enumerate(vocab)}
    counts = np.zeros((len(sequences), len(vocab)), float)
    for row, sequence in enumerate(sequences):
        for k in ks:
            for i in range(len(sequence) - k + 1):
                column = index.get(sequence[i:i + k])
                if column is not None:
                    counts[row, column] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-k", type=int, default=6, dest="max_k")
    parser.add_argument("--out", default="results/kmer_ladder.json")
    args = parser.parse_args()

    element_rows = pd.read_csv(ELEMENTS)
    sequences = elements().set_index("sequence_id").seq.loc[element_rows.sequence_id].values
    y, groups = element_rows.activity.values, element_rows.group.values

    probes = {}
    for name, path in REPRESENTATIONS:
        probes[name] = out_of_fold(np.load(path), y, groups, 5)
        print(f"{name}: rho {spearmanr(probes[name], y).statistic:+.4f}", flush=True)

    ladder = [(tuple(range(1, k + 1)), f"1-{k}-mer") for k in range(3, args.max_k + 1)]
    if args.max_k >= 6:
        ladder.append(((6,), "6-mer only"))

    results = []
    for ks, label in ladder:
        baseline = out_of_fold(kmer_matrix(sequences, ks), y, groups, 5)
        rho_base = float(spearmanr(baseline, y).statistic)
        width = sum(4 ** k for k in ks)
        print(f"\n{label} ({width} features): baseline rho {rho_base:+.4f}")
        for name, probe in probes.items():
            margin = float(spearmanr(probe, y).statistic - rho_base)
            low, high = paired_interval(probe, baseline, y, groups, 1000, 0)
            lows = [paired_interval(probe, baseline, y, groups, 1000, seed)[0]
                    for seed in range(VERDICT_SEEDS)]
            cleared = sum(1 for value in lows if value > 0)
            results.append({"baseline": label, "features": width, "baseline_rho": rho_base,
                            "representation": name, "margin": margin,
                            "low": float(low), "high": float(high),
                            "seeds_clearing": cleared, "seeds": VERDICT_SEEDS})
            verdict = "beats it" if cleared == VERDICT_SEEDS else (
                "loses to it" if cleared == 0 and margin < 0 else "level with it")
            print(f"    {name:<26} margin {margin:+.4f} [{low:+.4f}, {high:+.4f}]  "
                  f"{cleared}/{VERDICT_SEEDS} seeds, {verdict}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
