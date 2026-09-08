#!/usr/bin/env python3
"""Benchmark a model's single-variant effect predictions in regulatory sequence.

This is the primary benchmark. Each quartet contributes its two single-substitution
sequences, so the 2,833 pairs give 5,666 single variants, each with a measured log2
activity change relative to its own reference 200-mer and a standard error.

Why this target rather than the interaction contrast: the second difference is
mostly measurement noise. Its reliability is 0.29, capping any predictor at an
observed correlation of 0.535, and only 58 pairs carry the interaction flag. The
single-variant effect has reliability 0.73, a ceiling of 0.855, and 522 flagged
variants. The allele recoding does not enter here at all; it exists only to put the
four-haplotype contrast on the paper's convention, so every number below is on
plain ref/alt coding.

    python3 scripts/single_variant.py --predictions results/evo2_7b_base/predictions.csv \
        --out results/evo2_7b_base/single_variant.json --label "Evo 2 7B base"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_stats import auroc, element_kmers, gc_fraction, group_boot, noise_ceiling
from evo_probe import paired_interval
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from validation import PROTOCOL, group_splits, out_of_fold_linear

AUDIT = "data/audit.csv.gz"
# var1 always sits at position 100 and is the "altref" haplotype; var2 is downstream
# and is "refalt".  load_quartets orders the singles by position, so a maps to var1.
SOURCES = (("a", "altref"), ("b", "refalt"))


def single_variants(predictions_path, audit_path=AUDIT):
    """One row per single-substitution sequence, with its measured effect and SE."""
    pred = pd.read_csv(predictions_path)
    audit = pd.read_csv(audit_path)
    key = ["v1", "v2", "center_variant", "window", "library"]
    joined = audit[key[0]].astype(str)
    for column in key[1:]:
        joined = joined + ";" + audit[column].astype(str)
    audit = audit.assign(_k=joined)
    wanted = ["_k"] + [
        f"{name}_{field}" for _, name in SOURCES for field in ("log2Skew", "log2SkewSE", "emVar")
    ]
    merged = pred.assign(_k=pred.pair_id.str.split("|").str[0]).merge(
        audit.drop_duplicates("_k")[wanted], on="_k", how="left"
    )
    if len(merged) != len(pred):
        raise ValueError("audit join changed the row count")
    if not (merged.pos_a == 100).all():
        raise ValueError("expected the first variant at position 100 in every pair")

    frames = []
    for state, name in SOURCES:
        measured = merged[f"{name}_log2Skew"]
        if not np.allclose(merged[f"y_{state}"], measured, atol=1e-8, equal_nan=True):
            raise ValueError(
                f"y_{state} does not match {name}_log2Skew; the "
                "position ordering assumption is wrong"
            )
        frames.append(
            pd.DataFrame(
                {
                    "variant": state,
                    "pair_id": merged.pair_id,
                    "group_id": merged.group_id,
                    "sequence_id": merged[f"id_{state}"],
                    "seq": merged[f"seq_{state}"],
                    "seq_ref": merged.seq_wt,
                    "position": merged[f"pos_{state}"],
                    "y": merged[f"y_{state}"].astype(float),
                    "se": merged[f"{name}_log2SkewSE"].astype(float),
                    "emvar": merged[f"{name}_emVar"] == True,  # noqa: E712
                    "delta_score": (merged[f"s_{state}"] - merged.s_wt).astype(float),
                    "score_ref": merged.s_wt.astype(float),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    table = table[np.isfinite(table.y) & np.isfinite(table.se) & (table.se >= 0)]
    table = table.reset_index(drop=True)

    # A variant sequence can appear in more than one pair.  That is only safe for
    # grouped cross-validation if its copies stay inside one region group.
    spread = table.groupby("sequence_id").group_id.nunique()
    if (spread > 1).any():
        raise ValueError(
            f"{int((spread > 1).sum())} variant sequences span more "
            "than one region group; grouping would leak"
        )
    if table.empty:
        raise ValueError("no usable single variants")
    return table


def allele_features(table):
    """Reference and alternate base at the substituted position, one-hot."""
    ref = [s[p - 1] for s, p in zip(table.seq_ref, table.position)]
    alt = [s[p - 1] for s, p in zip(table.seq, table.position)]
    columns = []
    for bases, _ in ((ref, "ref"), (alt, "alt")):
        columns.append(np.stack([[b == base for b in bases] for base in "ACGT"], axis=1))
    return np.hstack(columns).astype(float)


def zero_shot(table, n_boot, seed):
    y, delta, groups = (
        table.y.to_numpy(),
        table.delta_score.to_numpy(),
        table.group_id.to_numpy(),
    )
    ceiling = noise_ceiling(y, table.se.to_numpy())
    limit = ceiling["perfect_predictor_observed_correlation_ceiling"]
    signed = float(spearmanr(delta, y).statistic)
    magnitude = float(spearmanr(np.abs(delta), np.abs(y)).statistic)
    return {
        "n": int(len(table)),
        "reliability": ceiling["reliability"],
        "ceiling": limit,
        "signed_spearman": signed,
        "signed_spearman_95ci": group_boot(
            groups, lambda i: spearmanr(delta[i], y[i]).statistic, n_boot, seed
        ),
        "signed_pearson": float(pearsonr(delta, y).statistic),
        "magnitude_spearman": magnitude,
        "magnitude_spearman_95ci": group_boot(
            groups,
            lambda i: spearmanr(np.abs(delta[i]), np.abs(y[i])).statistic,
            n_boot,
            seed,
        ),
        "disattenuated_signed": signed / limit if limit > 0 else None,
    }


def supervised(table, folds, seed):
    """Grouped out-of-fold prediction of the measured effect, model against baselines."""
    y, groups = table.y.to_numpy(), table.group_id.to_numpy()
    alt_kmers = element_kmers(table.seq.tolist())
    ref_kmers = element_kmers(table.seq_ref.tolist())
    features = {
        "model_delta_score": table.delta_score.to_numpy().reshape(-1, 1),
        "gc_content": gc_fraction(table.seq.tolist()).reshape(-1, 1),
        "position": table.position.to_numpy(float).reshape(-1, 1),
        "allele_identity": allele_features(table),
        "kmer_counts": alt_kmers,
        "kmer_delta": alt_kmers - ref_kmers,
        "kmer_delta_plus_alleles": np.hstack([alt_kmers - ref_kmers, allele_features(table)]),
    }
    features["kmer_delta_plus_model"] = np.hstack(
        [features["kmer_delta"], features["model_delta_score"]]
    )

    mean_prediction = np.empty(len(y))
    for train, test in group_splits(groups, folds, seed):
        mean_prediction[test] = y[train].mean()

    def scored(prediction):
        return {
            "spearman": float(spearmanr(prediction, y).statistic),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        }

    fitted = {
        name: out_of_fold_linear(x, y, groups, folds, ridge=x.shape[1] > 1, seed=seed)
        for name, x in features.items()
    }
    out = {name: scored(prediction) for name, prediction in fitted.items()}
    out["training_mean"] = scored(mean_prediction)
    out["perfect_predictor_rmse_floor"] = float(np.sqrt(np.mean(table.se.to_numpy() ** 2)))

    # The claim is the margin over the strongest cheap baseline, with its
    # uncertainty, not the level.  Resample whole region groups.
    reference = "kmer_delta"
    base_rho = out[reference]["spearman"]
    out["margin_over_kmer_delta"] = {
        name: {
            "margin": out[name]["spearman"] - base_rho,
            "interval": [
                float(v) for v in paired_interval(fitted[name], fitted[reference], y, groups)
            ],
        }
        for name in fitted
        if name != reference
    }
    return out


def detection(table, folds, seed, n_boot):
    """Can the model rank the variants the source study flagged as active-changing?"""
    label = table.emvar.to_numpy(bool)
    groups = table.group_id.to_numpy()
    if label.sum() == 0:
        return {"n_positives": 0}
    magnitude = np.abs(table.delta_score.to_numpy())
    gc = gc_fraction(table.seq.tolist())
    position = table.position.to_numpy(float)

    def out_of_fold_probability(x):
        probability = np.empty(len(label))
        for train, test in group_splits(groups, folds, seed):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            probability[test] = model.fit(x[train], label[train]).predict_proba(x[test])[:, 1]
        return probability

    covariates = np.column_stack([gc, position])
    without = out_of_fold_probability(covariates)
    with_model = out_of_fold_probability(np.column_stack([covariates, magnitude]))
    return {
        "n_positives": int(label.sum()),
        "positive_rate": float(label.mean()),
        "single_feature_auroc": {
            "model_delta_score_abs": auroc(magnitude, label),
            "gc_content": auroc(gc, label),
            "position": auroc(position, label),
        },
        "covariates_only": auroc(without, label),
        "covariates_plus_model": auroc(with_model, label),
        "gain": auroc(with_model, label) - auroc(without, label),
        "gain_95ci": group_boot(
            groups,
            lambda i: auroc(with_model[i], label[i]) - auroc(without[i], label[i]),
            n_boot,
            seed,
        ),
    }


def precision_strata(table, keeps=(1.0, 0.75, 0.5, 0.25)):
    rows = []
    for keep in keeps:
        subset = table[table.se <= table.se.quantile(keep)]
        ceiling = noise_ceiling(subset.y.to_numpy(), subset.se.to_numpy())
        limit = ceiling["perfect_predictor_observed_correlation_ceiling"]
        rho = float(spearmanr(subset.delta_score, subset.y).statistic)
        rows.append(
            {
                "keep": keep,
                "n": int(len(subset)),
                "reliability": ceiling["reliability"],
                "ceiling": limit,
                "signed_spearman": rho,
                "disattenuated": rho / limit if limit > 0 else None,
            }
        )
    return rows


def evaluate(predictions_path, out_path, label, audit=AUDIT, folds=5, seed=0, n_boot=1000):
    table = single_variants(predictions_path, audit)
    result = {
        "label": label,
        "target": "measured log2 activity change of one substitution, ref/alt coding",
        "note": "the allele recoding used for the interaction contrast does not apply here",
        "predictions": str(predictions_path),
        "zero_shot": zero_shot(table, n_boot, seed),
        "supervised_out_of_fold": supervised(table, folds, seed),
        "detection": detection(table, folds, seed, n_boot),
        "precision_strata": precision_strata(table),
        "protocol": PROTOCOL,
        "settings": {"folds": folds, "seed": seed, "bootstrap": n_boot},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    z, s, d = result["zero_shot"], result["supervised_out_of_fold"], result["detection"]
    print(f"\n{label}: {z['n']} single variants in {table.group_id.nunique()} region groups")
    print(f"  reliability {z['reliability']:.4f}, so no predictor can exceed {z['ceiling']:.4f}")
    print(
        f"  zero-shot signed Spearman    {z['signed_spearman']:+.4f}  "
        f"95% CI [{z['signed_spearman_95ci'][0]:+.4f}, {z['signed_spearman_95ci'][1]:+.4f}]"
    )
    print(
        f"  zero-shot magnitude Spearman {z['magnitude_spearman']:+.4f}  "
        f"95% CI [{z['magnitude_spearman_95ci'][0]:+.4f}, {z['magnitude_spearman_95ci'][1]:+.4f}]"
    )
    print("\n  grouped five-fold out-of-fold, predicting the measured effect")
    width = max(len(k) for k in s if k != "perfect_predictor_rmse_floor")
    for name, value in sorted(
        s.items(), key=lambda kv: -kv[1]["spearman"] if isinstance(kv[1], dict) else 1
    ):
        if isinstance(value, dict):
            print(f"    {name:{width}}  rho {value['spearman']:+.4f}   RMSE {value['rmse']:.4f}")
    print(
        f"    {'RMSE floor for a perfect predictor':{width}}       "
        f"       {s['perfect_predictor_rmse_floor']:.4f}"
    )
    if d.get("n_positives"):
        print(
            f"\n  detecting the {d['n_positives']} flagged variants "
            f"({100 * d['positive_rate']:.1f}% of rows)"
        )
        for name, value in d["single_feature_auroc"].items():
            print(f"    AUROC {name:24} {value:.4f}")
        print(
            f"    covariates only {d['covariates_only']:.4f}, plus the model "
            f"{d['covariates_plus_model']:.4f}, gain {d['gain']:+.5f} "
            f"95% CI [{d['gain_95ci'][0]:+.5f}, {d['gain_95ci'][1]:+.5f}]"
        )
    print(f"\nwrote {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--audit", default=AUDIT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    evaluate(
        args.predictions,
        args.out,
        args.label,
        args.audit,
        args.folds,
        args.seed,
        args.bootstrap,
    )


if __name__ == "__main__":
    main()
