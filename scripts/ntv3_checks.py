#!/usr/bin/env python3
"""Two checks on NTv3, nothing else.

  layer       probe one named layer from an existing sweep directory. CPU.
  likelihood  score the 2,595 reference elements and correlate. GPU.

    python3 scripts/ntv3_checks.py layer --embeddings results/ntv3_650m_sweep --layer 11
    python3 scripts/ntv3_checks.py likelihood --checkpoint InstaDeepAI/NTv3_650M_pre

The likelihood step scores only the distinct reference sequences, not all
10,856 quartet members, because the element-level correlation is what it is
for. That makes it about a quarter of the work of a full scoring run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import load_matrix
from element_data import elements, validate_elements
from evo_probe import kmers, paired_interval
from scipy.stats import spearmanr
from validation import mean_prediction, out_of_fold

MULTIPLE = 128


def layer(embeddings, layer, pooling="mean", folds=5):
    d = Path(embeddings)
    meta = json.loads((d / "meta.json").read_text())
    path = d / f"X_{pooling}_L{layer}.npy"
    if not path.exists():
        path = d / f"X_{pooling}.npy"
        recorded = str(meta.get("layer"))
        if recorded not in (str(layer), f"core.transformer_blocks.{layer}.final_layer_norm"):
            raise ValueError(
                "single-layer matrix does not identify the requested transformer layer"
            )
        if not path.exists():
            found = sorted(int(f.stem.split("_L")[1]) for f in d.glob(f"X_{pooling}_L*.npy"))
            raise SystemExit(f"no matrix for layer {layer} in {d}. Available: {found}")
    el = pd.read_csv(d / "elements.csv")
    X = load_matrix(path, el, ["sequence_id"])
    full = elements().set_index("sequence_id")
    validate_elements(el, full)
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values

    gc = np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs])
    km = kmers(seqs)
    print(
        f"{meta.get('checkpoint', d)}   layer {layer}   pooling {pooling}   "
        f"n={len(el)}   width {X.shape[1]}\n"
    )

    preds, rows = {}, []
    for name, feat in [
        ("GC content (1 feature)", gc),
        ("1-2-3 k-mer counts (84 features)", km),
        (f"NTv3 layer {layer} (probe)", X),
        (f"NTv3 layer {layer} + k-mers", np.hstack([X, km])),
    ]:
        preds[name] = out_of_fold(feat, y, g, folds)
        rows.append(
            (
                name,
                spearmanr(preds[name], y).statistic,
                float(np.sqrt(np.mean((preds[name] - y) ** 2))),
            )
        )
    mean = mean_prediction(y, g, folds)
    rows.append(
        (
            "predict the mean",
            float(spearmanr(mean, y).statistic),
            float(np.sqrt(np.mean((y - mean) ** 2))),
        )
    )

    width = max(len(r[0]) for r in rows)
    print(f"{'':{width}}   Spearman     RMSE")
    for name, rho, rmse in rows:
        print(f"{name:{width}}   {rho:+.4f}   {rmse:.4f}")

    a, b = f"NTv3 layer {layer} (probe)", "1-2-3 k-mer counts (84 features)"
    margin = spearmanr(preds[a], y).statistic - spearmanr(preds[b], y).statistic
    low, high = paired_interval(preds[a], preds[b], y, g)
    print(f"\nprobe minus k-mers: {margin:+.4f}  95% interval [{low:+.4f}, {high:+.4f}]")
    print(
        "Beats k-mer counts."
        if low > 0
        else "Reliably WORSE than k-mer counts."
        if high < 0
        else "Not distinguishable from k-mer counts on this evidence."
    )


def likelihood(
    checkpoint="InstaDeepAI/NTv3_650M_pre",
    revision="main",
    out="results/ntv3_650m_elements",
    batch_size=100,
):
    from benchmark_data import reverse_complement
    from ntv3_score import NTv3Scorer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    el = elements()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorer = NTv3Scorer(checkpoint, revision, use_bfloat16=False)
    scorer.POSITION_CHUNK = batch_size
    device = scorer.device
    seqs = el.seq.tolist()
    length = len(seqs[0])
    score = scorer._pseudo_log_likelihood
    complement = reverse_complement
    print(f"{checkpoint}: scoring {len(seqs)} reference elements on {device}", flush=True)
    scores = []
    for i, sequence in enumerate(seqs):
        scores.append((score(sequence) + score(complement(sequence))) / 2)
        if i % 100 == 0:
            print(f"  {i}/{len(seqs)}", flush=True)

    el = el.assign(likelihood=scores)
    el.drop(columns=["seq"]).to_csv(out / "element_likelihood.csv", index=False)
    rho = spearmanr(el.likelihood, el.activity).statistic
    active = el[el.active.astype(str).str.lower().isin(("true", "1"))]
    print(f"\n{checkpoint} likelihood vs element activity")
    print(f"  Spearman            {rho:+.4f}   (n={len(el)})")
    print(
        f"  active elements only {spearmanr(active.likelihood, active.activity).statistic:+.4f}"
        f"   (n={len(active)})"
    )
    print(f"  per base            {np.mean(scores) / length:+.4f}")
    print(f"\nwrote {out}/element_likelihood.csv")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("layer", help="CPU. Probe one named layer.")
    a.add_argument("--embeddings", default="results/ntv3_650m_sweep")
    a.add_argument("--layer", type=int, required=True)
    a.add_argument("--pooling", choices=("mean", "last"), default="mean")
    a.add_argument("--folds", type=int, default=5)

    b = sub.add_parser("likelihood", help="GPU. Score the reference elements.")
    b.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_pre")
    b.add_argument("--revision", default="main")
    b.add_argument("--out", default="results/ntv3_650m_elements")
    b.add_argument("--batch-size", type=int, default=100)

    args = vars(parser.parse_args())
    cmd = args.pop("cmd")
    (layer if cmd == "layer" else likelihood)(**args)


if __name__ == "__main__":
    main()
