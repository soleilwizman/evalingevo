#!/usr/bin/env python3
"""Two-stage analysis: sequence-score diagnostics, then whole-element ridge probes.

Raw likelihood is not in MPRA units. Its association is reported directly;
activity-scale RMSE is reported only for predictions calibrated out of fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import aligned_tables, atomic_json, load_matrix, source_hashes
from benchmark_data import file_hash
from benchmark_models import MODELS
from benchmark_stats import correlations, element_kmers, gc_fraction, group_boot
from element_data import elements, validate_elements
from model_runtime import POOLING_PROTOCOL
from single_variant import single_variants
from validation import PROTOCOL, mean_prediction, out_of_fold, out_of_fold_linear


def metrics(prediction, y, groups, n_boot, seed, *, activity_scale):
    prediction, y = np.asarray(prediction), np.asarray(y)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("predictions must be finite and aligned with targets")
    result = correlations(y, prediction)
    if result["spearman"] is not None:

        def statistic(take):
            value = correlations(y[take], prediction[take])["spearman"]
            return np.nan if value is None else value

        result["spearman_95ci"] = group_boot(groups, statistic, n_boot, seed)
    else:
        result["spearman_95ci"] = None
    result["rmse"] = float(np.sqrt(np.mean((prediction - y) ** 2))) if activity_scale else None
    result["rmse_units"] = "MPRA log2 activity/effect" if activity_scale else None
    if not activity_scale:
        result["rmse_note"] = "not defined across raw likelihood and MPRA units; use calibrated row"
    return result


def baseline_features(table, task):
    if task == "element":
        return {"gc": gc_fraction(table.seq).reshape(-1, 1), "kmer": element_kmers(table.seq)}
    return {
        "gc": gc_fraction(table.seq).reshape(-1, 1),
        "kmer": element_kmers(table.seq) - element_kmers(table.seq_ref),
    }


def fitted_readouts(features, y, groups, folds, seed, n_boot):
    predictions = {name: out_of_fold(X, y, groups, folds, seed) for name, X in features.items()}
    predictions["training_mean"] = mean_prediction(y, groups, folds, seed)
    rows = {
        name: metrics(p, y, groups, n_boot, seed, activity_scale=True)
        for name, p in predictions.items()
    }
    for name, X in features.items():
        rows[name]["features"] = int(X.shape[1])
        rows[name]["estimator"] = "nested grouped ridge with fold-local scaling"
    return predictions, rows


def gains(prediction, baselines, y, groups, n_boot, seed):
    result = {}
    for name in ("gc", "kmer"):
        baseline = baselines[name]
        rho = correlations(y, prediction)["spearman"]
        base_rho = correlations(y, baseline)["spearman"]

        def rmse_gain(take):
            return np.sqrt(np.mean((baseline[take] - y[take]) ** 2)) - np.sqrt(
                np.mean((prediction[take] - y[take]) ** 2)
            )

        result[name] = {
            "spearman_gain": rho - base_rho if rho is not None and base_rho is not None else None,
            "rmse_reduction": float(rmse_gain(np.arange(len(y)))),
            "rmse_reduction_95ci": group_boot(groups, rmse_gain, n_boot, seed),
        }
    return result


def score_analysis(results_root="results", audit="data/audit.csv.gz", folds=5, seed=0, n_boot=1000):
    root = Path(results_root)
    source = root / MODELS["evo2"].score_directory / "predictions.csv"
    reference = {"element": elements(source, audit), "variant": single_variants(source, audit)}
    tasks = {}
    for task, table in reference.items():
        target, group = ("activity", "group") if task == "element" else ("y", "group_id")
        y, groups = table[target].to_numpy(), table[group].to_numpy()
        baseline_predictions, baseline_rows = fitted_readouts(
            baseline_features(table, task), y, groups, folds, seed, n_boot
        )
        models = {}
        for key, spec in MODELS.items():
            path = root / spec.score_directory / "predictions.csv"
            if not path.exists():
                models[key] = {"label": spec.label, "status": "not_computed", "missing": str(path)}
                continue
            observed = elements(path, audit) if task == "element" else single_variants(path, audit)
            columns = (
                ["sequence_id", "seq", "activity", "group"]
                if task == "element"
                else ["pair_id", "variant", "sequence_id", "seq_ref", "position", "y", "group_id"]
            )
            aligned_tables(table, observed, columns, numeric_columns=(target,))
            scores = (
                observed.s_wt.to_numpy() if task == "element" else observed.delta_score.to_numpy()
            )
            calibrated = out_of_fold_linear(scores, y, groups, folds, seed=seed)
            models[key] = {
                "label": spec.label,
                "status": "computed",
                "source": str(path),
                "source_sha256": file_hash(path),
                "raw_score": metrics(scores, y, groups, n_boot, seed, activity_scale=False),
                "calibrated_score": metrics(
                    calibrated, y, groups, n_boot, seed, activity_scale=True
                ),
                "calibration": "one-feature linear regression fitted within each outer training fold",
                "calibrated_gains_over_baselines": gains(
                    calibrated, baseline_predictions, y, groups, n_boot, seed
                ),
            }
        tasks[task] = {
            "n": len(y),
            "groups": int(len(np.unique(groups))),
            "target": "reference log2(RNA/DNA) activity"
            if task == "element"
            else "signed log2 activity change, alt minus reference",
            "baselines": baseline_rows,
            "models": models,
        }
    return {"stage": "score_diagnostics", "tasks": tasks}


def primary_embedding(directory, table, spec):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    spec.check_revision(meta.get("revision"))
    expected = {
        "checkpoint": spec.checkpoint,
        "representation": spec.representation,
        "layer": str(spec.layer),
        "primary_pooling": "mean",
        "pooling_protocol": "native-real-token-v2"
        if spec.family == "dnabert2"
        else POOLING_PROTOCOL,
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(
                f"{directory}: expected {field}={value!r}; no substitute layer is allowed"
            )
    rows = pd.read_csv(directory / "elements.csv")
    validate_elements(rows, table.set_index("sequence_id"))
    matrix = load_matrix(directory / "X_mean.npy", rows, ["sequence_id"])
    positions = pd.Index(rows.sequence_id).get_indexer(table.sequence_id)
    if (positions < 0).any() or len(rows) != len(table):
        raise ValueError(f"{directory}: primary probes require the complete shared element table")
    return matrix[positions], meta


def probe_analysis(
    embeddings_root="results/v2",
    pred="results/evo2_7b_base/predictions.csv",
    audit="data/audit.csv.gz",
    folds=5,
    seed=0,
    n_boot=1000,
):
    table = elements(pred, audit)
    y, groups = table.activity.to_numpy(), table.group.to_numpy()
    features = baseline_features(table, "element")
    metadata, missing = {}, {}
    for key, spec in MODELS.items():
        path = Path(embeddings_root) / f"{key}_elements"
        if not path.exists():
            missing[key] = {
                "label": spec.label,
                "status": "not_computed",
                "missing": str(path),
                "required_representation": spec.representation,
            }
        else:
            features[key], metadata[key] = primary_embedding(path, table, spec)
    predictions, rows = fitted_readouts(features, y, groups, folds, seed, n_boot)
    models = dict(missing)
    for key, meta in metadata.items():
        models[key] = {
            "label": MODELS[key].label,
            "status": "computed",
            "metrics": rows[key],
            "embedding_metadata": meta,
            "gains_over_baselines": gains(predictions[key], predictions, y, groups, n_boot, seed),
        }
    return {
        "stage": "whole_element_probes",
        "n": len(y),
        "groups": int(len(np.unique(groups))),
        "target": "reference log2(RNA/DNA) activity",
        "pooling": "mean over real native units",
        "baselines": {key: rows[key] for key in ("gc", "kmer", "training_mean")},
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("scores", "probes"):
        command = sub.add_parser(stage)
        command.add_argument("--out", required=True)
        command.add_argument("--audit", default="data/audit.csv.gz")
        command.add_argument("--folds", type=int, default=5)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--bootstrap", dest="n_boot", type=int, default=1000)
        if stage == "scores":
            command.add_argument("--results-root", default="results")
        else:
            command.add_argument("--embeddings-root", default="results/v2")
            command.add_argument("--pred", default="results/evo2_7b_base/predictions.csv")
    args = vars(parser.parse_args())
    stage, out = args.pop("stage"), args.pop("out")
    if args["n_boot"] < 1:
        raise ValueError("bootstrap count must be positive")
    result = (score_analysis if stage == "scores" else probe_analysis)(**args)
    result.update(protocol=PROTOCOL, settings=args, source_sha256=source_hashes())
    atomic_json(out, result)
    print(f"wrote {out}; missing models are explicit gaps, not substitute checkpoints/layers")


if __name__ == "__main__":
    main()
