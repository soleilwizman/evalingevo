"""Diagnostics for the allele recoding: what it changes, and how much of it is noise.

The recoding (Siraj et al.) makes the lowest-activity diplotype the baseline.  On the
four-haplotype contrast that reduces to one +1/-1 per pair, applied once in
``evo_epistasis.add_flip``.  This script answers three questions the README needs:

1. What does recoding actually change?  Only the mean, never the magnitude.
2. Which measurement-error ceiling goes with which coding?
3. How much of the recoded positive shift survives when the true interaction is zero?

Run:  python3 scripts/recoding_bias.py --quartets data/quartets.csv.gz --audit data/audit.csv.gz \
           --predictions results/evo2_7b_base/predictions.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

STATES = ("wt", "a", "b", "ab")
SOURCE_STATES = ("refref", "altref", "refalt", "altalt")


def recode_flip(y):
    """+1 when the least-active corner sits on the WT/AB diagonal, -1 on the A/B diagonal."""
    return np.where(np.isin(np.argmin(y, axis=1), (0, 3)), 1.0, -1.0)


def coding_comparison(epsilon, flip, epsilon_se):
    """Recoding cannot change |epsilon|, so E[eps^2] is identical and only the mean moves."""
    recoded = flip * epsilon
    noise = float((epsilon_se**2).mean())
    rows = []
    for name, value in (("ref/alt", epsilon), ("recoded", recoded)):
        variance = float(np.var(value))
        rows.append(
            {
                "coding": name,
                "mean_squared_epsilon": float((value**2).mean()),
                "mean": float(value.mean()),
                "variance": variance,
                "mean_squared_se": noise,
                "signal_variance": variance - noise,
                "reliability": (variance - noise) / variance,
                "ceiling": float(np.sqrt(max((variance - noise) / variance, 0.0))),
                "fraction_positive": float((value > 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    identity = {
        "variance_gap": rows[0]["variance"] - rows[1]["variance"],
        "squared_mean_gap": rows[1]["mean"] ** 2 - rows[0]["mean"] ** 2,
        "rmse_floor": float(np.sqrt(noise)),
        "rmse_zero_baseline": float(np.sqrt((recoded**2).mean())),
        "rmse_mean_baseline_analytic": float(np.sqrt(np.var(recoded))),
    }
    return table, identity


def paired_correlations(predictions):
    """Each rho against the ceiling estimated on its own coding."""
    df = pd.read_csv(predictions)
    model_refalt, model_recoded = df.flip * df.model_interaction, df.model_interaction
    return pd.DataFrame(
        [
            {
                "coding": "ref/alt",
                "spearman": spearmanr(model_refalt, df.epsilon_refalt).statistic,
                "pearson": pearsonr(model_refalt, df.epsilon_refalt).statistic,
            },
            {
                "coding": "recoded",
                "spearman": spearmanr(model_recoded, df.epsilon).statistic,
                "pearson": pearsonr(model_recoded, df.epsilon).statistic,
            },
        ]
    )


def join_source_errors(quartets, audit_path):
    """Attach the four per-diplotype activity readings and their standard errors."""
    audit = pd.read_csv(audit_path)
    audit = audit[audit.status == "accepted"].copy()
    key = ["v1", "v2", "center_variant", "window", "library"]
    audit["_k"] = audit[key].astype(str).agg(";".join, axis=1)
    quartets = quartets.copy()
    quartets["_k"] = quartets.pair_id.str.split("|").str[0]
    columns = (
        ["_k"] + [f"{s}_Log2FC" for s in SOURCE_STATES] + [f"{s}_Log2FC_SE" for s in SOURCE_STATES]
    )
    merged = quartets.merge(audit.drop_duplicates("_k")[columns], on="_k", how="left")
    if len(merged) != len(quartets) or merged[columns[1:]].isna().any().any():
        raise SystemExit("audit join is incomplete; cannot run the null")
    return merged


def null_simulation(merged, draws=25, seeds=8, calibrate=True):
    """Recode a world with no interaction at all, and see how positive epsilon still looks.

    True double activity is forced to the additive sum of its singles, so the true
    interaction is exactly zero for every pair.  All four activities are then redrawn
    from their published standard errors and recoded by the same lowest-corner rule.
    Observed activities stand in for the truth, which widens the real spread between
    corners and makes the baseline easier to identify, so this understates the bias.
    """
    activity = merged[[f"{s}_Log2FC" for s in SOURCE_STATES]].to_numpy(float)
    se = merged[[f"{s}_Log2FC_SE" for s in SOURCE_STATES]].to_numpy(float)
    if calibrate:
        independent = np.sqrt((se**2).sum(1))
        se = se * (merged.epsilon_se.to_numpy(float) / independent)[:, None]
    truth = activity.copy()
    truth[:, 3] = activity[:, 1] + activity[:, 2] - activity[:, 0]
    means, positive = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        batch_mean, batch_positive = [], []
        for _ in range(draws):
            drawn = truth + rng.normal(0.0, 1.0, truth.shape) * se
            epsilon = (
                (drawn[:, 1] - drawn[:, 0])
                + (drawn[:, 2] - drawn[:, 0])
                - (drawn[:, 3] - drawn[:, 0])
            )
            recoded = recode_flip(drawn) * epsilon
            batch_mean.append(recoded.mean())
            batch_positive.append((recoded > 0).mean())
        means.append(float(np.mean(batch_mean)))
        positive.append(float(np.mean(batch_positive)))
    return {
        "mean": float(np.mean(means)),
        "mean_range": [min(means), max(means)],
        "fraction_positive": float(np.mean(positive)),
        "fraction_positive_range": [min(positive), max(positive)],
    }


def observed_from_source(merged):
    """Same statistic on the unperturbed source readings, as the comparison point."""
    activity = merged[[f"{s}_Log2FC" for s in SOURCE_STATES]].to_numpy(float)
    epsilon = (
        (activity[:, 1] - activity[:, 0])
        + (activity[:, 2] - activity[:, 0])
        - (activity[:, 3] - activity[:, 0])
    )
    recoded = recode_flip(activity) * epsilon
    return {
        "mean": float(recoded.mean()),
        "fraction_positive": float((recoded > 0).mean()),
        "agreement_with_published_epsilon": float(np.corrcoef(epsilon, merged.epsilon)[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quartets", default="data/quartets.csv.gz")
    parser.add_argument("--audit", default="data/audit.csv.gz")
    parser.add_argument("--predictions", default="results/evo2_7b_base/predictions.csv")
    parser.add_argument("--draws", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    quartets = pd.read_csv(args.quartets)
    y = quartets[[f"y_{s}" for s in STATES]].to_numpy(float)
    flip = recode_flip(y)
    table, identity = coding_comparison(
        quartets.epsilon.to_numpy(float), flip, quartets.epsilon_se.to_numpy(float)
    )

    print("== 1. what recoding changes ==")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(
        f"\n  variance gap {identity['variance_gap']:.5f} "
        f"= squared-mean gap {identity['squared_mean_gap']:.5f}   (recoding moves the mean, not the spread)"
    )
    print(
        f"  RMSE floor {identity['rmse_floor']:.5f}   zero baseline {identity['rmse_zero_baseline']:.5f}"
        f"   mean baseline {identity['rmse_mean_baseline_analytic']:.5f}"
    )
    print(
        f"  competitive range above the mean baseline "
        f"{identity['rmse_mean_baseline_analytic'] - identity['rmse_floor']:.5f}"
    )

    print("\n== 2. each correlation against the ceiling for its own coding ==")
    print(
        paired_correlations(args.predictions).to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"
        )
    )

    print("\n== 3. null: true interaction exactly zero for every pair ==")
    merged = join_source_errors(quartets, args.audit)
    observed = observed_from_source(merged)
    print(
        f"  source readings reproduce the published epsilon at r={observed['agreement_with_published_epsilon']:.4f}"
    )
    print(
        f"  observed                      mean {observed['mean']:+.4f}   positive {observed['fraction_positive']:.4f}"
    )
    for label, calibrate in (
        ("errors calibrated to epsilon_se", True),
        ("raw per-state errors", False),
    ):
        null = null_simulation(merged, args.draws, args.seeds, calibrate)
        print(
            f"  null, {label:<32} mean {null['mean']:+.4f} "
            f"[{null['mean_range'][0]:+.4f},{null['mean_range'][1]:+.4f}]"
            f"   positive {null['fraction_positive']:.4f} "
            f"[{null['fraction_positive_range'][0]:.4f},{null['fraction_positive_range'][1]:.4f}]"
        )


if __name__ == "__main__":
    main()
