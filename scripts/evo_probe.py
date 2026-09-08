import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import load_matrix, source_hashes
from benchmark_data import file_hash
from benchmark_stats import element_kmers
from element_data import elements as elements
from element_data import validate_elements
from scipy.stats import spearmanr
from validation import mean_prediction, permutation_null
from validation import out_of_fold as out_of_fold

PRED = "results/evo2_7b_base/predictions.csv"
AUDIT = "data/audit.csv.gz"
ALPHAS = np.logspace(-2, 6, 25)
POOLINGS = ("mean", "last")


# --------------------------------------------------------------------------
# the element table: one row per distinct reference 200-mer, with its activity
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# step 1 -- GPU. Save two summaries of one hidden layer per sequence.
# --------------------------------------------------------------------------
def embed(
    out,
    layer="blocks.26.mlp.l3",
    checkpoint="evo2_7b_base",
    weights=None,
    batch_size=4,
    pred=PRED,
    audit=AUDIT,
    device="cuda:0",
):
    from model_embeddings import Evo2Embedder
    from model_runtime import POOLING_PROTOCOL

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    el = elements(pred, audit)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    embedder = Evo2Embedder(checkpoint, layer, weights, device)
    pooled = {p: [] for p in POOLINGS}
    for start in range(0, len(el), batch_size):
        values = embedder(el.seq.iloc[start : start + batch_size].tolist())
        for name, block in zip(POOLINGS, values):
            pooled[name].append(block)
        if start % (batch_size * 50) == 0:
            print(f"  {start}/{len(el)}", flush=True)
    for name, chunks in pooled.items():
        X = np.concatenate(chunks).astype(np.float32)
        np.save(out / f"X_{name}.npy", X)
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "layer": layer,
                "checkpoint": checkpoint,
                "poolings": list(POOLINGS),
                "n": len(el),
                "width": int(X.shape[1]),
                "pooling_protocol": POOLING_PROTOCOL,
                "device": embedder.device,
                "batch_size": batch_size,
                "weights_sha256": file_hash(weights) if weights else None,
                "source_sha256": source_hashes(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}/X_mean.npy and X_last.npy, width {X.shape[1]}")


# --------------------------------------------------------------------------
# step 2 -- no GPU.
# --------------------------------------------------------------------------
def kmers(seqs, k_max=3):
    """Counts of every DNA word up to k_max. Occurrences OVERLAP, so AAAA
    contains three AAs; str.count would say two."""
    return element_kmers(seqs, k_max)


VERDICT_MARGIN, VERDICT_SEEDS = 0.01, 10


def report_margin(pred_a, pred_b, y, groups, margin, win, tie, n_boot=1000):
    """Print the interval on rho(a) - rho(b), refusing a verdict on the boundary."""
    low, high = paired_interval(pred_a, pred_b, y, groups, n_boot, 0)
    print(f"probe minus word counts: {margin:+.4f}  95% interval [{low:+.4f}, {high:+.4f}]")
    if abs(low) < VERDICT_MARGIN:
        lows = [low] + [
            paired_interval(pred_a, pred_b, y, groups, n_boot, s)[0]
            for s in range(1, VERDICT_SEEDS)
        ]
        cleared = sum(1 for value in lows if value > 0)
        print(
            f"lower bound over {VERDICT_SEEDS} seeds: min {min(lows):+.4f}, "
            f"max {max(lows):+.4f}, clears zero {cleared}/{VERDICT_SEEDS}"
        )
        if cleared not in (0, VERDICT_SEEDS):
            print(
                "On the boundary: the verdict flips with the bootstrap seed. Report the "
                "margin and this spread, not a pass/fail."
            )
            return low, high
    print(win if low > 0 else tie)
    return low, high


def paired_interval(pred_a, pred_b, y, groups, n_boot=1000, seed=0):
    """Bootstrap whole groups; return the 95% interval on rho(a) - rho(b)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n_boot):
        take = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq), True)])
        diffs.append(
            spearmanr(pred_a[take], y[take]).statistic - spearmanr(pred_b[take], y[take]).statistic
        )
    return np.nanpercentile(diffs, [2.5, 97.5])


def probe(embeddings, pooling="mean", pred=PRED, audit=AUDIT, folds=5, n_permutations=20):
    d = Path(embeddings)
    el = pd.read_csv(d / "elements.csv")
    X = load_matrix(d / f"X_{pooling}.npy", el, ["sequence_id"])
    full = elements(pred, audit).set_index("sequence_id")
    validate_elements(el, full)
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values
    meta = json.loads((d / "meta.json").read_text())
    print(f"n={len(el)}  layer {meta['layer']}  pooling {pooling}  width {X.shape[1]}\n")

    gc = np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs])
    km = kmers(seqs)

    probe_name = f"Evo hidden layer {meta['layer']} (probe)"
    preds, rows = {}, []
    for name, feat in [
        ("Evo score (1 feature)", el[["s_wt"]].values),
        ("GC content (1 feature)", gc),
        ("DNA word counts (84 features)", km),
        (probe_name, X),
        (f"{probe_name} + word counts", np.hstack([X, km])),
    ]:
        preds[name] = out_of_fold(feat, y, g, folds)
        rows.append(
            (
                name,
                spearmanr(preds[name], y).statistic,
                float(np.sqrt(np.mean((preds[name] - y) ** 2))),
            )
        )
    rows.append(
        (
            "predict the mean",
            float(spearmanr(mean_prediction(y, g, folds), y).statistic),
            float(np.sqrt(np.mean((y - mean_prediction(y, g, folds)) ** 2))),
        )
    )

    w = max(len(r[0]) for r in rows)
    print(f"{'':{w}}   Spearman     RMSE")
    for name, rho, rmse in rows:
        print(f"{name:{w}}   {rho:+.4f}   {rmse:.4f}")

    # How big does a number have to be before it means anything here?
    null = permutation_null(X, y, g, folds, draws=n_permutations)
    print(
        f"\nnoise floor from {n_permutations} label permutations: "
        f"{np.mean(null):+.4f} +/- {np.std(null):.4f}"
    )

    a, b = probe_name, "DNA word counts (84 features)"
    got = dict((r[0], r[1]) for r in rows)
    print()
    report_margin(
        preds[a],
        preds[b],
        y,
        g,
        got[a] - got[b],
        "The information is in there, and it beats word counts.",
        "No evidence Evo's representations add anything over word counts.",
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="GPU: pool one hidden layer per sequence")
    e.add_argument("--out", default="results/embeddings")
    e.add_argument("--layer", default="blocks.26.mlp.l3")
    e.add_argument("--checkpoint", default="evo2_7b_base")
    e.add_argument("--weights", default=None)
    e.add_argument("--batch-size", type=int, default=4)

    p = sub.add_parser("probe", help="no GPU: out-of-fold ridge against baselines")
    p.add_argument("--embeddings", default="results/embeddings")
    p.add_argument("--pooling", default="mean", choices=POOLINGS)
    p.add_argument("--folds", type=int, default=5)

    args = {k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()}
    cmd = args.pop("cmd")
    (embed if cmd == "embed" else probe)(**args)


if __name__ == "__main__":
    main()
