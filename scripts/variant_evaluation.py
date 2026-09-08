"""Evaluate variant representations against matched sequence baselines."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import aligned_tables, load_matrix
from evo_probe import kmers, paired_interval
from scipy.stats import spearmanr
from validation import (
    PROTOCOL,
    mean_prediction,
    out_of_fold,
    permutation_null,
    selected_predictions,
)
from variant_data import noise_ceiling, within_element_accuracy

SEED = 0


def build_features(directory, obs):
    d = Path(directory)
    seqs = pd.read_csv(d / "sequences.csv")
    keys = ["wt_id", "mut_id", "index"]
    aligned_tables(obs, seqs, keys)
    position = np.column_stack([obs["index"].values, (obs.side == "a").astype(int).values])
    return {
        "variant position + side (2)": position,
        "k-mer difference (84)": kmers(seqs.mut_seq.values) - kmers(seqs.wt_seq.values),
        "reference embedding only (control)": load_matrix(d / "X_wt_mean.npy", obs, keys),
        "difference, mean pooled": load_matrix(d / "X_d_mean.npy", obs, keys),
        "difference, at the variant unit": load_matrix(d / "X_d_pos.npy", obs, keys),
    }


def score_one(embeddings, target="magnitude", folds=5, seed=SEED, n_permutations=20, quiet=False):
    d = Path(embeddings)
    obs = pd.read_csv(d / "observations.csv")
    if target not in ("magnitude", "signed"):
        raise ValueError("target must be magnitude or signed")
    meta = json.loads((d / "meta.json").read_text())
    y = np.abs(obs.effect.values) if target == "magnitude" else obs.effect.values
    g = obs.group.values
    features = build_features(d, obs)

    preds, rows = {}, []
    for name, X in features.items():
        preds[name] = out_of_fold(X, y, g, folds, seed)
        accuracy, n_pairs = within_element_accuracy(preds[name], y, obs.wt_id.values)
        rows.append(
            (
                name,
                float(spearmanr(preds[name], y).statistic),
                float(np.sqrt(np.mean((preds[name] - y) ** 2))),
                accuracy,
            )
        )
    rows.append(
        (
            "predict the mean",
            float(spearmanr(mean_prediction(y, g, folds, seed), y).statistic),
            float(np.sqrt(np.mean((y - mean_prediction(y, g, folds, seed)) ** 2))),
            0.5,
        )
    )
    selected_name = "difference, inner-selected pooling"
    selected, selections = selected_predictions(
        {
            name: features[name]
            for name in ("difference, mean pooled", "difference, at the variant unit")
        },
        y,
        g,
        folds,
        seed,
    )
    preds[selected_name] = selected
    accuracy, n_pairs = within_element_accuracy(selected, y, obs.wt_id.values)
    rows.append(
        (
            selected_name,
            float(spearmanr(selected, y).statistic),
            float(np.sqrt(np.mean((selected - y) ** 2))),
            accuracy,
        )
    )
    named = {r[0]: r for r in rows}
    best = named[selected_name]

    ceiling = noise_ceiling(obs) if target == "signed" else None
    if not quiet:
        print(f"n={len(obs)}  {meta['label']}  width {meta['width']}  target {target}")
        print(
            f"pooled over {meta.get('units', 'base')}s, "
            f"{meta.get('units_per_sequence', 200):.1f} per sequence, "
            f"grouped {folds}-fold seed {seed}"
        )
        if ceiling:
            print(
                f"measurement reliability {ceiling['reliability']:.4f}, so no predictor "
                f"of the signed effect can exceed {ceiling['ceiling']:.4f}"
            )
        elif target == "magnitude":
            print(
                "no ceiling for the magnitude target: attenuation is a linear-model "
                "result and |y| is not linear. Run with --target signed for it."
            )
        print()
        _, n_pairs = within_element_accuracy(preds[best[0]], y, obs.wt_id.values)
        width = max(len(r[0]) for r in rows)
        print(f"{'':{width}}   Spearman     RMSE   within-element")
        for name, rho, rmse, accuracy in rows:
            print(f"{name:{width}}   {rho:+.4f}   {rmse:.4f}   {accuracy:.3f}")
        print(
            f"\nwithin-element column: {n_pairs} pairs sharing an element, ranked "
            "on which variant matters more. Chance is 0.500 and knowing the "
            "element cannot help, so this is the variant question on its own."
        )
        # Fix the candidate before permutation; the selected comparison above
        # already nests feature selection inside the training groups.
        null = permutation_null(
            features["difference, mean pooled"], y, g, folds, seed, n_permutations
        )
        if null:
            print(
                f"\nmean-pooled probe block-permutation null: {np.mean(null):+.4f} +/- {np.std(null):.4f}"
            )

    rivals = [
        named["variant position + side (2)"],
        named["k-mer difference (84)"],
        named["reference embedding only (control)"],
    ]
    intervals, beaten = {}, []
    for rival in rivals:
        low, high = paired_interval(preds[best[0]], preds[rival[0]], y, g)
        intervals[rival[0]] = (best[1] - rival[1], low, high)
        beaten.append(low > 0)
        if not quiet:
            print(
                f"\n{best[0]} minus {rival[0]}: {best[1] - rival[1]:+.4f}  "
                f"95% interval [{low:+.4f}, {high:+.4f}]"
            )

    control_low = intervals["reference embedding only (control)"][1]
    verdict = (
        "locates the variant"
        if all(beaten)
        else "no better than the element control"
        if control_low <= 0
        else "beats the control but not every baseline"
    )
    if not quiet:
        print(f"\n{verdict}.")
    return {
        "protocol": PROTOCOL,
        "selected_per_fold": selections,
        "label": meta["label"],
        "units": meta.get("units", "base"),
        "ceiling": ceiling,
        "rows": rows,
        "best": best[0],
        "spearman": best[1],
        "intervals": intervals,
        "verdict": verdict,
    }


def probe(embeddings, target="magnitude", folds=5, seed=SEED):
    score_one(embeddings, target, folds, seed)


def compare(embeddings, target="magnitude", folds=5, seed=SEED):
    # The baselines are printed once, from the first run, so every run must be
    # over the same rows. Different n means different observations, and the
    # shared baseline line would silently describe only one of them.
    # Count the rows on disk rather than trusting meta.json, which records what
    # the embed run intended rather than what is actually there.
    tables = [pd.read_csv(Path(d) / "observations.csv") for d in embeddings]
    if not tables:
        raise ValueError("at least one embedding directory is required")
    for table in tables[1:]:
        aligned_tables(tables[0], table, ["wt_id", "mut_id", "index", "group", "side", "effect"])
    counts = {d: len(table) for d, table in zip(embeddings, tables)}
    results = [score_one(d, target, folds, seed, quiet=True) for d in embeddings]
    print(f"target {target}, grouped {folds}-fold seed {seed}, n={next(iter(counts.values()))}\n")
    print("baselines, identical for every model:")
    for name, rho, _, _ in results[0]["rows"]:
        if name in (
            "variant position + side (2)",
            "k-mer difference (84)",
            "predict the mean",
        ):
            print(f"  {name:<36} {rho:+.4f}")
    width = max(len(r["label"]) for r in results)
    print(f"\n{'model':{width}}  units  feature  probe    within   control   probe minus control")
    for r in results:
        by_name = dict((x[0], x) for x in r["rows"])
        control = by_name["reference embedding only (control)"][1]
        within = by_name[r["best"]][3]
        gap, low, high = r["intervals"]["reference embedding only (control)"]
        short = "selected"
        print(
            f"{r['label']:{width}}  {r['units']:<5}  {short}  {r['spearman']:+.4f}  "
            f"{within:.3f}   {control:+.4f}   {gap:+.4f} [{low:+.4f}, {high:+.4f}]  "
            f"{r['verdict']}"
        )
