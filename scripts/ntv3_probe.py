import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import load_matrix, source_hashes
from element_data import elements, validate_elements
from evo_probe import AUDIT, PRED, kmers, report_margin
from model_runtime import pad_to_multiple as pad_to_multiple
from scipy.stats import spearmanr
from validation import mean_prediction, out_of_fold, permutation_null

MULTIPLE = 128
DEFAULT_CHECKPOINT = "InstaDeepAI/NTv3_100M_pre"


def token_offset(tokenizer, multiple=MULTIPLE):
    """Where sequence tokens start, so pooling covers the real bases only."""
    probe_seq = "ACGT" * (multiple // 4)
    ids = tokenizer(probe_seq, add_special_tokens=True)["input_ids"]
    wanted = tokenizer.convert_tokens_to_ids(list("ACGT"))
    for start in range(len(ids) - len(probe_seq) + 1):
        if list(ids[start : start + 4]) == list(wanted) and len(ids) - start >= len(probe_seq):
            return start
    raise ValueError(f"cannot align tokens to bases: {len(ids)} ids for {len(probe_seq)} bases")


def embed(
    out,
    layer=11,
    checkpoint=DEFAULT_CHECKPOINT,
    revision="main",
    pred=PRED,
    audit=AUDIT,
    batch_size=8,
    device="auto",
):
    from model_embeddings import NTv3Embedder
    from model_runtime import POOLING_PROTOCOL, pad_to_multiple

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    el = elements(pred, audit)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    embedder = NTv3Embedder(checkpoint, layer, revision, device)
    rows = {"mean": [], "last": []}
    for start in range(0, len(el), batch_size):
        for name, block in zip(rows, embedder(el.seq.iloc[start : start + batch_size].tolist())):
            rows[name].append(block)
        if start % (batch_size * 25) == 0:
            print(f"  {start}/{len(el)}", flush=True)
    matrices = {name: np.concatenate(chunks) for name, chunks in rows.items()}
    for name, matrix in matrices.items():
        np.save(out / f"X_{name}.npy", matrix)
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model": "ntv3",
                "checkpoint": checkpoint,
                "revision": revision,
                "layer": embedder.representation,
                "width": int(matrices["mean"].shape[1]),
                "n": len(el),
                "padded_length": len(pad_to_multiple(el.seq.iloc[0])[0]),
                "pooling_protocol": POOLING_PROTOCOL,
                "pooled_over_real_bases_only": True,
                "device": embedder.device,
                "batch_size": batch_size,
                "source_sha256": source_hashes(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}/X_mean.npy and X_last.npy, width {matrices['mean'].shape[1]}")


def probe(embeddings, pooling="mean", pred=PRED, audit=AUDIT, folds=5, n_permutations=20):
    d = Path(embeddings)
    el = pd.read_csv(d / "elements.csv")
    X = load_matrix(d / f"X_{pooling}.npy", el, ["sequence_id"])
    full = elements(pred, audit).set_index("sequence_id")
    validate_elements(el, full)
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values
    meta = json.loads((d / "meta.json").read_text())
    print(
        f"n={len(el)}  {meta['checkpoint']}  layer {meta['layer']}  "
        f"pooling {pooling}  width {X.shape[1]}\n"
    )

    gc = np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs])
    km = kmers(seqs)

    preds, rows = {}, []
    for name, feat in [
        ("GC content (1 feature)", gc),
        ("DNA word counts (84 features)", km),
        ("NTv3 hidden layer (probe)", X),
        ("NTv3 hidden layer + word counts", np.hstack([X, km])),
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

    width = max(len(r[0]) for r in rows)
    print(f"{'':{width}}   Spearman     RMSE")
    for name, rho, rmse in rows:
        print(f"{name:{width}}   {rho:+.4f}   {rmse:.4f}")

    shuffled = permutation_null(X, y, g, folds, draws=n_permutations)
    print(
        f"\nnoise floor from {n_permutations} label permutations: "
        f"{np.mean(shuffled):+.4f} +/- {np.std(shuffled):.4f}"
    )

    margin = (
        spearmanr(preds["NTv3 hidden layer (probe)"], y).statistic
        - spearmanr(preds["DNA word counts (84 features)"], y).statistic
    )
    report_margin(
        preds["NTv3 hidden layer (probe)"],
        preds["DNA word counts (84 features)"],
        y,
        g,
        margin,
        "The information is in there, and it beats word counts.",
        "Not distinguishable from word counts on this evidence.",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="GPU. Pool one hidden layer per element.")
    e.add_argument("--out", required=True)
    e.add_argument(
        "--layer",
        type=int,
        default=11,
        help="actual NTv3 transformer block index (default 11)",
    )
    e.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    e.add_argument("--revision", default="main")
    e.add_argument("--batch-size", type=int, default=8)

    p = sub.add_parser("probe", help="CPU. Fit ridge on the saved embeddings.")
    p.add_argument("--embeddings", default="results/ntv3_100m_final")
    p.add_argument("--pooling", choices=("mean", "last"), default="mean")
    p.add_argument("--folds", type=int, default=5)

    args = vars(parser.parse_args())
    cmd = args.pop("cmd")
    (embed if cmd == "embed" else probe)(**args)


if __name__ == "__main__":
    main()
