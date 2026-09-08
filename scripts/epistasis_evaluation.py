"""Grouped statistical evaluation of scored regulatory quartets."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_data import CONTRAST, KMER_VOCAB, STATES, file_hash, load_quartets
from benchmark_plots import make_element_plot, make_plot
from benchmark_stats import (
    auroc,
    correlations,
    element_kmers,
    gc_fraction,
    group_boot,
    metrics,
    noise_ceiling,
)
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from validation import PROTOCOL, group_splits
from validation import out_of_fold_linear as out_of_fold_linear


def sequence_only_features(quartets):
    """Build sequence/design features without using measured activity values."""
    features = np.zeros((len(quartets), len(STATES) * len(KMER_VOCAB) + 3), float)
    for row_index, row in enumerate(quartets.itertuples(index=False)):
        offset = 0
        for state in STATES:
            sequence = getattr(row, f"seq_{state}")
            for kmer_index, kmer in enumerate(KMER_VOCAB):
                features[row_index, offset + kmer_index] = sum(
                    sequence[i : i + len(kmer)] == kmer
                    for i in range(len(sequence) - len(kmer) + 1)
                )
            offset += len(KMER_VOCAB)
        features[row_index, -3:] = (row.pos_a, row.pos_b, row.distance)
    return features


def add_predictions(quartets, scores, folds=5, seed=0):
    if scores.sequence_id.duplicated().any():
        raise ValueError("Duplicate sequence scores")
    df, mapping = quartets.copy(), scores.set_index("sequence_id").score
    for state in STATES:
        df[f"s_{state}"] = df[f"id_{state}"].map(mapping)
    if not np.isfinite(df[[f"s_{s}" for s in STATES]].to_numpy()).all():
        raise ValueError("Missing or nonfinite sequence score")
    df["model_interaction"] = df.s_a + df.s_b - df.s_wt - df.s_ab
    df = add_flip(df)
    if not 2 <= folds <= df.group_id.nunique():
        raise ValueError("Invalid grouped-fold count")
    y = df.epsilon.to_numpy()
    features = sequence_only_features(df)
    df["additive_zero"], df["fold"] = 0.0, -1
    names = (
        "calibrated_model",
        "training_mean",
        "training_median",
        "sequence_only_kmer_ridge",
        "majority_sign",
    )
    for name in names:
        df[name] = np.nan
    coefficients = []
    for fold, (train, test) in enumerate(group_splits(df.group_id, folds, seed)):
        model = LinearRegression().fit(df.model_interaction.to_numpy()[train, None], y[train])
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features[train], y[train])
        df.loc[df.index[test], "calibrated_model"] = model.predict(
            df.model_interaction.to_numpy()[test, None]
        )
        df.loc[df.index[test], "sequence_only_kmer_ridge"] = ridge.predict(features[test])
        df.loc[df.index[test], "training_mean"] = y[train].mean()
        df.loc[df.index[test], "training_median"] = np.median(y[train])
        df.loc[df.index[test], "majority_sign"] = (
            1.0 if np.sum(y[train] > 0) >= np.sum(y[train] < 0) else -1.0
        )
        df.loc[df.index[test], "fold"] = fold
        coefficients.append(
            {
                "fold": fold,
                "intercept": float(model.intercept_),
                "slope": float(model.coef_[0]),
            }
        )
    return df, coefficients


def cluster_intervals(df, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(df.group_id.to_numpy() == g) for g in df.group_id.unique()]
    samples = {
        k: []
        for k in (
            "spearman",
            "pearson",
            "rmse_gain_zero",
            "rmse_gain_mean",
            "rmse_gain_sequence",
        )
    }
    y, raw, calibrated = (
        df[c].to_numpy() for c in ("epsilon", "model_interaction", "calibrated_model")
    )
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        for key, value in correlations(y[idx], raw[idx]).items():
            if value is not None:
                samples[key].append(value)
        model_rmse = np.sqrt(np.mean((y[idx] - calibrated[idx]) ** 2))
        for key, column in (
            ("rmse_gain_zero", "additive_zero"),
            ("rmse_gain_mean", "training_mean"),
            ("rmse_gain_sequence", "sequence_only_kmer_ridge"),
        ):
            samples[key].append(
                float(np.sqrt(np.mean((y[idx] - df[column].to_numpy()[idx]) ** 2)) - model_rmse)
            )
    return {k: np.quantile(v, [0.025, 0.975]).tolist() if v else None for k, v in samples.items()}


def select_cases(df, n=3, threshold=0.25):
    eligible = df[df.epsilon.abs() >= threshold].copy()
    eligible["correct_sign"] = np.sign(eligible.epsilon) == np.sign(eligible.model_interaction)
    eligible["oof_abs_error"] = (eligible.epsilon - eligible.calibrated_model).abs()
    success = (
        eligible[eligible.correct_sign]
        .sort_values(["oof_abs_error", "pair_id"])
        .head(n)
        .assign(case="success")
    )
    failure = (
        eligible[~eligible.correct_sign]
        .sort_values(["epsilon", "pair_id"], key=lambda x: -x.abs() if x.name == "epsilon" else x)
        .head(n)
        .assign(case="failure")
    )
    control = (
        df[df.epsilon.abs() < threshold]
        .sort_values("epsilon", key=lambda x: x.abs())
        .head(n)
        .assign(case="near_additive")
    )
    return pd.concat([success, failure, control], ignore_index=True)


def add_flip(df):
    """Apply the paper's allele recoding to the measured and model contrasts.

    Siraj et al. recode alleles lowest-to-highest activity and treat the
    lowest-activity diplotype as the reference category.  The four diplotypes
    pair into complements (wt<->ab, a<->b), so this can only flip the sign of
    the second difference; magnitude is untouched.  The identical per-pair flip
    is applied to the model contrast, since flipping the measurement alone
    would impose a random per-pair sign and destroy the comparison.
    ``epsilon_refalt`` keeps the original ref/alt contrast, which is what the
    noise ceiling is estimated on.
    """
    y = df[[f"y_{s}" for s in STATES]].to_numpy(float)
    flip = np.where(np.isin(np.argmin(y, axis=1), (0, 3)), 1.0, -1.0)
    df = df.copy()
    df["flip"] = flip
    df["epsilon_refalt"] = df["epsilon"]
    df["epsilon"] = flip * df["epsilon"]
    df["model_interaction"] = flip * df["model_interaction"]
    return df


def join_audit(df, audit_path):
    audit = pd.read_csv(audit_path)
    key = ["v1", "v2", "center_variant", "window", "library"]
    if any(c not in audit.columns for c in key):
        raise ValueError(
            f"audit is missing join columns: {[c for c in key if c not in audit.columns]}"
        )
    joined = audit[key[0]].astype(str)
    for column in key[1:]:
        joined = joined + ";" + audit[column].astype(str)
    audit = audit.assign(_k=joined)
    wanted = [
        c for c in ("_k", "refref_Log2FC", "refref_active", "int_emVar") if c in audit.columns
    ]
    merged = df.assign(_k=df.pair_id.str.split("|").str[0]).merge(
        audit.drop_duplicates("_k")[wanted], on="_k", how="left"
    )
    if len(merged) != len(df):
        raise ValueError("audit join changed the row count")
    return merged


def reproduction_check(df):
    labelled = df[df.int_emVar == True]  # noqa: E712
    return {
        "n_pairs": int(len(df)),
        "n_labelled": int(len(labelled)),
        "flip_negative": int((df.flip < 0).sum()),
        "dampening_labelled_refalt": float((labelled.epsilon_refalt > 0).mean()),
        "dampening_labelled_recoded": float((labelled.epsilon > 0).mean()),
        "dampening_all_recoded": float((df.epsilon > 0).mean()),
        "paper_dampening_fraction": 139 / 180,
    }


def single_variant_report(df, n_boot=1000, seed=0):
    measured = np.concatenate([df.y_a.to_numpy(float), df.y_b.to_numpy(float)])
    model = np.concatenate([(df.s_a - df.s_wt).to_numpy(float), (df.s_b - df.s_wt).to_numpy(float)])
    groups = np.concatenate([df.group_id.to_numpy(), df.group_id.to_numpy()])
    interval = group_boot(
        groups,
        lambda i: spearmanr(np.abs(model[i]), np.abs(measured[i])).statistic,
        n_boot,
        seed,
    )
    return {
        "n": int(len(measured)),
        "magnitude_spearman": float(spearmanr(np.abs(model), np.abs(measured)).statistic),
        "magnitude_95ci": interval,
        "signed_spearman": float(spearmanr(model, measured).statistic),
        "sd_model_single_effect": float(np.std(model, ddof=1)),
        "sd_whole_sequence_score": float(df.s_wt.std()),
        "scrambled_join_would_give": float(df.s_wt.std() * np.sqrt(2)),
    }


def element_report(df, folds=5, n_boot=1000, seed=0, plot_path=None, label="Evo 2"):
    elements = (
        df.dropna(subset=["refref_Log2FC"])
        .drop_duplicates("seq_wt")[["seq_wt", "s_wt", "refref_Log2FC", "refref_active", "group_id"]]
        .reset_index(drop=True)
    )
    if elements.groupby("seq_wt").group_id.nunique().gt(1).any():
        raise ValueError("a reference sequence spans more than one cross-validation group")
    y = elements.refref_Log2FC.to_numpy(float)
    groups, model = elements.group_id.to_numpy(), elements.s_wt.to_numpy(float)
    gc, kmers = gc_fraction(elements.seq_wt), element_kmers(elements.seq_wt)
    active = elements.refref_active.astype(str).str.lower().isin(("true", "1")).to_numpy()
    mean_prediction = np.empty(len(y))
    for train, test in group_splits(groups, folds, seed):
        mean_prediction[test] = y[train].mean()

    def scored(prediction):
        return {
            "spearman": float(spearmanr(prediction, y).statistic),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        }

    fitted = {
        "gc": out_of_fold_linear(gc, y, groups, folds, seed=seed),
        "model": out_of_fold_linear(model, y, groups, folds, seed=seed),
        "kmer_1_2_3": out_of_fold_linear(kmers, y, groups, folds, ridge=True, seed=seed),
        "training_mean": mean_prediction,
    }
    plot = (
        make_element_plot(elements, y, gc, model, fitted, groups, plot_path, n_boot, seed, label)
        if plot_path
        else None
    )

    return {
        "n_elements": int(len(elements)),
        "n_active": int(active.sum()),
        "plot": plot,
        "raw_spearman": {
            "gc_vs_activity": float(spearmanr(gc, y).statistic),
            "gc_95ci": group_boot(groups, lambda i: spearmanr(gc[i], y[i]).statistic, n_boot, seed),
            "model_vs_activity": float(spearmanr(model, y).statistic),
            "model_95ci": group_boot(
                groups, lambda i: spearmanr(model[i], y[i]).statistic, n_boot, seed
            ),
            "model_vs_activity_active_only": float(spearmanr(model[active], y[active]).statistic),
            "model_vs_gc": float(spearmanr(model, gc).statistic),
        },
        "out_of_fold": {name: scored(value) for name, value in fitted.items()},
    }


def detection_report(df, folds=5, n_boot=1000, seed=0, n_seeds=20):
    label = (df.int_emVar == True).to_numpy()  # noqa: E712
    groups = df.group_id.to_numpy()
    gc = gc_fraction(df.seq_wt)
    single = (df.y_a.abs() + df.y_b.abs()).to_numpy(float)
    closeness = -df.distance.to_numpy(float)
    model = np.abs(df.model_interaction.to_numpy(float))

    def out_of_fold_probability(features, split_seed=None):
        probability = np.empty(len(label))
        split_seed = seed if split_seed is None else split_seed
        for train, test in group_splits(groups, folds, split_seed):
            fitted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            fitted.fit(features[train], label[train])
            probability[test] = fitted.predict_proba(features[test])[:, 1]
        return probability

    covariates = np.column_stack([closeness, single, gc])
    with_evo = np.column_stack([covariates, model])
    without = out_of_fold_probability(covariates)
    with_model = out_of_fold_probability(with_evo)

    # The absolute AUROC moves by a few points with the fold draw, so the gain is
    # re-estimated over several splits and reported with its spread.  The claim is
    # the gain, not the level.
    across = []
    for other in range(n_seeds):
        a = auroc(out_of_fold_probability(covariates, other), label)
        b = auroc(out_of_fold_probability(with_evo, other), label)
        across.append(
            {
                "seed": other,
                "covariates_only": a,
                "covariates_plus_model": b,
                "gain": b - a,
            }
        )
    gains = np.array([row["gain"] for row in across])
    levels = np.array([row["covariates_only"] for row in across])

    return {
        "n": int(len(df)),
        "n_positives": int(label.sum()),
        "across_seeds": {
            "n_seeds": n_seeds,
            "covariates_only_mean": float(levels.mean()),
            "covariates_only_range": [float(levels.min()), float(levels.max())],
            "gain_mean": float(gains.mean()),
            "gain_range": [float(gains.min()), float(gains.max())],
            "per_seed": across,
        },
        "single_feature_auroc": {
            "single_variant_effect_size": auroc(single, label),
            "model_interaction_abs": auroc(model, label),
            "closeness_negative_bp": auroc(closeness, label),
            "gc_content": auroc(gc, label),
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


def precision_strata(df, keeps=(1.0, 0.75, 0.5, 0.25)):
    rows = []
    for keep in keeps:
        subset = df[df.epsilon_se <= df.epsilon_se.quantile(keep)]
        ceiling = noise_ceiling(subset.epsilon_refalt, subset.epsilon_se)
        rho = float(spearmanr(subset.model_interaction, subset.epsilon).statistic)
        ceiling_value = (
            ceiling["max_observed_correlation"]
            if "max_observed_correlation" in ceiling
            else float(np.sqrt(max(ceiling["reliability"], 0.0)))
        )
        rows.append(
            {
                "keep": keep,
                "n": int(len(subset)),
                "reliability": ceiling["reliability"],
                "ceiling": ceiling_value,
                "spearman": rho,
                "disattenuated": rho / ceiling_value if ceiling_value > 0 else None,
            }
        )
    return rows


def audit_analyses(df, audit_path, folds=5, n_boot=1000, seed=0, plot_path=None, label="Evo 2"):
    """Everything that needs the paper's labels and raw element activity."""
    joined = join_audit(df, audit_path)
    return {
        "audit": str(audit_path),
        "reproduction": reproduction_check(joined),
        "single_variants": single_variant_report(joined, n_boot, seed),
        "elements": element_report(joined, folds, n_boot, seed, plot_path, label),
        "detection": detection_report(joined, folds, n_boot, seed),
        "precision_strata": precision_strata(joined),
        "libraries": joined.pair_id.str.split("|")
        .str[0]
        .str.split(";")
        .str[-1]
        .value_counts()
        .to_dict(),
    }


def evaluate(
    quartets_path,
    scores_path,
    out,
    label="Evo 2",
    folds=5,
    seed=0,
    n_boot=1000,
    threshold=0.25,
    audit=None,
):
    if n_boot < 1 or threshold < 0:
        raise ValueError("bootstrap must be positive and threshold nonnegative")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, coefficients = add_predictions(
        load_quartets(quartets_path), pd.read_csv(scores_path), folds, seed
    )
    df.to_csv(out / "predictions.csv", index=False)
    select_cases(df, threshold=threshold).to_csv(out / "cases.csv", index=False)
    strata = make_plot(df, out / "plots.png", label)
    y = df.epsilon.to_numpy()
    prediction_metrics = {
        x: metrics(y, df[x], threshold, True)
        for x in (
            "calibrated_model",
            "additive_zero",
            "training_mean",
            "training_median",
            "sequence_only_kmer_ridge",
        )
    }
    table = pd.DataFrame(
        [
            {"method": "raw " + label, **metrics(y, df.model_interaction, threshold)},
            *[{"method": name, **values} for name, values in prediction_metrics.items()],
        ]
    )
    table.to_csv(out / "results_table.csv", index=False)
    result = {
        "label": label,
        "contrast": CONTRAST,
        "pairs": len(df),
        "groups": int(df.group_id.nunique()),
        "raw": metrics(y, df.model_interaction, threshold),
        "raw_all_nonzero_signs": metrics(y, df.model_interaction, 0),
        "predictions": prediction_metrics,
        "majority_sign": metrics(y, df.majority_sign, threshold),
        "calibration": coefficients,
        "cluster_bootstrap_95ci": cluster_intervals(df, n_boot, seed),
        "strata": strata,
        "protocol": PROTOCOL,
        "settings": {
            "folds": folds,
            "seed": seed,
            "bootstrap": n_boot,
            "sign_threshold": threshold,
        },
        "inputs": {
            "quartets_sha256": file_hash(quartets_path),
            "scores_sha256": file_hash(scores_path),
        },
    }
    if "epsilon_se" in df:
        se = pd.to_numeric(df.epsilon_se, errors="coerce")
        result["noise_ceiling"] = noise_ceiling(df.epsilon_refalt, se)
        reliability = result["noise_ceiling"]["reliability"]
        if reliability > 0:
            scale = np.sqrt(reliability)
            ci = result["cluster_bootstrap_95ci"]
            result["noise_ceiling"].update(
                {
                    "spearman_reliability_adjusted_95ci_approx": [
                        float(np.clip(value / scale, -1.0, 1.0)) for value in ci["spearman"]
                    ],
                    "spearman_upper_95_reliability_adjusted_approx": float(
                        np.clip(ci["spearman"][1] / scale, -1.0, 1.0)
                    ),
                    "pearson_reliability_adjusted_95ci": [
                        float(np.clip(value / scale, -1.0, 1.0)) for value in ci["pearson"]
                    ],
                    "pearson_upper_95_reliability_adjusted": float(
                        np.clip(ci["pearson"][1] / scale, -1.0, 1.0)
                    ),
                    "correlation_adjustment_note": "Spearman adjustment is approximate; classical attenuation correction is defined for Pearson correlation.",
                }
            )
        mask = np.isfinite(se) & (se > 0) & (df.epsilon.abs() > 1.96 * se)
        result["exploratory_source_SE_subset"] = metrics(
            df.epsilon[mask], df.model_interaction[mask], threshold
        )
    if audit and Path(audit).exists():
        result["audit_analyses"] = audit_analyses(
            df, audit, folds, n_boot, seed, out / "elements_gc_kmer.png", label
        )
    elif audit:
        result["audit_analyses"] = f"skipped, {audit} not found"
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
