#!/usr/bin/env python3
"""Embed whole reference elements at the prespecified cross-model readouts."""

import argparse
from pathlib import Path

import numpy as np
from artifact_io import atomic_json, source_hashes
from benchmark_data import file_hash
from benchmark_models import MODELS
from element_data import AUDIT, PRED, elements
from model_adapters import DNABERT2Adapter, Evo2Adapter, NTv3Adapter
from model_runtime import POOLING_PROTOCOL

ADAPTERS = {"evo2": Evo2Adapter, "ntv3": NTv3Adapter, "dnabert2": DNABERT2Adapter}


def embed(model_key, out, revision=None, weights=None, pred=PRED, audit=AUDIT, limit=0):
    spec = MODELS[model_key]
    spec.check_revision(revision)
    if weights and spec.family != "evo2":
        raise ValueError("--weights is supported only for Evo 2")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory is not empty; keep historical and smoke runs separate")
    table = elements(pred, audit)
    if limit:
        table = table.head(limit)
    kwargs = {"checkpoint": spec.checkpoint, "layer": spec.layer}
    kwargs.update({"weights": weights} if spec.family == "evo2" else {"revision": revision})
    adapter = ADAPTERS[spec.family](**kwargs)
    means, lasts = [], []
    for i, sequence in enumerate(table.seq):
        hidden, _ = adapter.encode(sequence)
        means.append(hidden.mean(0).cpu().numpy())
        lasts.append(hidden[-1].cpu().numpy())
        if i % 250 == 0:
            print(f"{spec.label}: {i}/{len(table)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("mean", means), ("last", lasts)):
        matrix = np.stack(rows).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite element embeddings")
        np.save(out / f"X_{name}.npy", matrix)
    table.drop(columns="seq").to_csv(out / "elements.csv", index=False)
    atomic_json(
        out / "meta.json",
        {
            "model": spec.family,
            "model_key": model_key,
            "checkpoint": spec.checkpoint,
            "layer": str(spec.layer),
            "representation": spec.representation,
            "resolved_layer": adapter.label,
            "revision": revision,
            "poolings": ["mean", "last"],
            "primary_pooling": "mean",
            "pooling_protocol": "native-real-token-v2"
            if spec.family == "dnabert2"
            else POOLING_PROTOCOL,
            "units": adapter.units,
            "n": len(table),
            "width": int(matrix.shape[1]),
            "limit": limit,
            "weights_sha256": file_hash(weights) if weights else None,
            "predictions_sha256": file_hash(pred),
            "audit_sha256": file_hash(audit),
            "source_sha256": source_hashes(),
        },
    )
    print(f"wrote {out}: {spec.representation}, mean over real {adapter.units}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--weights")
    parser.add_argument("--pred", default=PRED)
    parser.add_argument("--audit", default=AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    args = vars(parser.parse_args())
    args["model_key"] = args.pop("model")
    embed(**args)


if __name__ == "__main__":
    main()
