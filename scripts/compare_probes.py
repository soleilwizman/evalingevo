#!/usr/bin/env python3
"""Test whether two embedding probes are actually separable on the element task.

A bar chart invites the reader to rank two bars that differ by 0.003. This
answers whether that ranking is real: it refits both probes on the same element
table under the same folds, then resamples whole region groups for a 95%
interval on the difference. It also refits one probe under several fold draws,
so the reader can see how much the number moves for no reason at all.

    python3 scripts/compare_probes.py --a results/ntv3_100m_final --b results/ntv3_650m_final \
        --label-a "NTv3 100M probe" --label-b "NTv3 650M probe" \
        --out results/probe_comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import aligned_tables, load_matrix
from evo_probe import paired_interval
from scipy.stats import spearmanr
from validation import PROTOCOL, out_of_fold


def seeded_out_of_fold(x, y, groups, seed, folds=5):
    return float(spearmanr(out_of_fold(x, y, groups, folds, seed), y).statistic)


def compare(dir_a, dir_b, label_a, label_b, pooling="mean", folds=5, draws=6):
    a = pd.read_csv(Path(dir_a) / "elements.csv")
    b = pd.read_csv(Path(dir_b) / "elements.csv")
    aligned_tables(a, b, ["sequence_id", "activity", "group"])
    xa = load_matrix(Path(dir_a) / f"X_{pooling}.npy", a, ["sequence_id"])
    xb = load_matrix(Path(dir_b) / f"X_{pooling}.npy", b, ["sequence_id"])
    if draws < 1:
        raise ValueError("draws must be positive")
    y, groups = a.activity.values, a.group.values

    pa, pb = out_of_fold(xa, y, groups, folds), out_of_fold(xb, y, groups, folds)
    ra, rb = float(spearmanr(pa, y).statistic), float(spearmanr(pb, y).statistic)
    low, high = (float(v) for v in paired_interval(pa, pb, y, groups))
    spread = [seeded_out_of_fold(xa, y, groups, seed, folds) for seed in range(draws)]

    verdict = (
        f"{label_b} is reliably worse"
        if low > 0
        else f"{label_a} is reliably worse"
        if high < 0
        else "not distinguishable: the interval contains zero"
    )
    return {
        "protocol": PROTOCOL,
        "n": int(len(a)),
        "pooling": pooling,
        "folds": folds,
        "a": {
            "label": label_a,
            "dir": str(dir_a),
            "width": int(xa.shape[1]),
            "spearman": ra,
        },
        "b": {
            "label": label_b,
            "dir": str(dir_b),
            "width": int(xb.shape[1]),
            "spearman": rb,
        },
        "difference": ra - rb,
        "difference_95ci": [low, high],
        "verdict": verdict,
        "fold_draw_sensitivity": {
            "probe": label_a,
            "n_draws": draws,
            "min": float(np.min(spread)),
            "max": float(np.max(spread)),
            "range": float(np.ptp(spread)),
            "note": "the same probe refit under different fold shuffles; a gap "
            "smaller than this range is not a result",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--pooling", default="mean", choices=("mean", "last"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(
        args.a, args.b, args.label_a, args.label_b, args.pooling, args.folds, args.draws
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{result['a']['label']} {result['a']['spearman']:+.4f}  "
        f"{result['b']['label']} {result['b']['spearman']:+.4f}"
    )
    print(
        f"difference {result['difference']:+.4f}  95% CI "
        f"[{result['difference_95ci'][0]:+.4f}, {result['difference_95ci'][1]:+.4f}]"
    )
    print(result["verdict"])
    s = result["fold_draw_sensitivity"]
    print(
        f"same probe across {s['n_draws']} fold draws: {s['min']:+.4f} to {s['max']:+.4f} "
        f"(range {s['range']:.4f})"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
