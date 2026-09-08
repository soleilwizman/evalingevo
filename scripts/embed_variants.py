#!/usr/bin/env python3
"""GPU: embed the 5,428 variant sequences, in the same layout as every other
embedding directory in this repo.

The committed embeddings cover the 2,595 reference 200-mers only, so no probe
of a variant effect is possible. This adds the missing half. It embeds the
variant sequences and nothing else; the references already exist, and
variant_probe.py checks the two directories name the same checkpoint, layer and
revision before it subtracts one from the other.

    # Evo 2 (needs the evo2 package and an NVIDIA GPU)
    python3 embed_variants.py --backend evo2 --out results/evo_variants \
        --layer blocks.26.mlp.l3 --batch-size 4

    # NTv3 (needs a Hugging Face login for the gated InstaDeepAI checkpoints)
    python3 embed_variants.py --backend ntv3 --out results/ntv3_100m_variants \
        --checkpoint InstaDeepAI/NTv3_100M_pre --layer 5 --batch-size 16
    python3 embed_variants.py --backend ntv3 --out results/ntv3_650m_variants \
        --checkpoint InstaDeepAI/NTv3_650M_pre --layer 11 --batch-size 8

The run is resumable. Rows are written straight into a memory-mapped .npy and
progress.json records how many are done, so an interrupted run picks up where
it stopped. Re-running with a different configuration refuses rather than
mixing two checkpoints into one matrix.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import atomic_json, load_matrix, source_hashes
from benchmark_data import file_hash, load_quartets
from model_embeddings import Evo2Embedder as Evo2Embedder
from model_embeddings import NTv3Embedder as NTv3Embedder
from model_runtime import POOLING_PROTOCOL

MULTIPLE = 128
POOLINGS = ("mean", "last")


def sequence_table(quartets_path, include_reference=False, include_double=False):
    """One row per distinct sequence. Variants only by default."""
    quartets = load_quartets(quartets_path)
    wanted = (
        (("wt",) if include_reference else ()) + ("a", "b") + (("ab",) if include_double else ())
    )
    frames = []
    for state in wanted:
        frames.append(
            pd.DataFrame(
                {
                    "sequence_id": quartets[f"id_{state}"],
                    "seq": quartets[f"seq_{state}"],
                    "role": state,
                    "group_id": quartets.group_id,
                    "pair_id": quartets.pair_id,
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    # A sequence can appear in several pairs; embed it once.
    table = (
        table.sort_values(["sequence_id", "pair_id"])
        .drop_duplicates("sequence_id")
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )
    if table.seq.str.len().nunique() != 1:
        raise SystemExit("sequences are not all the same length")
    if table.sequence_id.duplicated().any():
        raise SystemExit("duplicate sequence ids after dedup")
    return table


def run(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = sequence_table(args.quartets, args.include_reference, args.include_double)
    counts = table.role.value_counts().to_dict()
    print(f"{len(table)} distinct sequences to embed  {counts}", flush=True)

    meta_path = out / "meta.json"
    meta = {
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "layer": str(args.layer),
        "revision": args.revision,
        "n": int(len(table)),
        "include_reference": bool(args.include_reference),
        "include_double": bool(args.include_double),
        "representation": (
            str(args.layer)
            if args.backend == "evo2"
            else f"core.transformer_blocks.{int(args.layer)}.final_layer_norm"
        ),
        "quartets_sha256": file_hash(args.quartets),
        "code_sha256": file_hash(__file__),
        "source_sha256": source_hashes(),
        "pooling_protocol": POOLING_PROTOCOL,
        "batch_size": args.batch_size,
        "weights_sha256": file_hash(args.weights) if args.weights else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        drift = {k for k in meta if k not in ("platform", "python") and existing.get(k) != meta[k]}
        if drift:
            raise SystemExit(
                f"this directory holds a different run; fields differ: "
                f"{sorted(drift)}. Use a new --out."
            )
        meta = {**existing, **meta}
    elif (out / "progress.json").exists() or any(out.glob("X_*.npy")):
        raise ValueError("embedding cache has no provenance; use a new output directory")
    atomic_json(meta_path, meta)

    progress_path = out / "progress.json"
    done = json.loads(progress_path.read_text())["done"] if progress_path.exists() else 0
    if not isinstance(done, int) or not 0 <= done <= len(table):
        raise ValueError("invalid embedding progress")
    if done == len(table):
        for name in POOLINGS:
            load_matrix(out / f"X_{name}.npy", table, ["sequence_id"])
        if not (out / "sequences.csv").exists() or "width" not in meta:
            raise ValueError("completed cache is missing its row table or width metadata")
        print("already complete", flush=True)
        return
    if done:
        for name in POOLINGS:
            if not (out / f"X_{name}.npy").exists():
                raise ValueError("partial cache is missing an embedding matrix")
        print(f"resuming at row {done}", flush=True)

    embedder = (
        Evo2Embedder(args.checkpoint, args.layer, args.weights)
        if args.backend == "evo2"
        else NTv3Embedder(args.checkpoint, int(args.layer), args.revision)
    )

    memmaps, sequences = {}, table.seq.tolist()
    for start in range(done, len(table), args.batch_size):
        batch = sequences[start : start + args.batch_size]
        pooled = dict(zip(POOLINGS, embedder(batch)))
        for name, block in pooled.items():
            path = out / f"X_{name}.npy"
            if name not in memmaps:
                if path.exists():
                    memmaps[name] = np.lib.format.open_memmap(path, mode="r+")
                    if memmaps[name].shape != (len(table), block.shape[1]):
                        raise SystemExit(
                            f"{path} has shape {memmaps[name].shape}, "
                            f"expected {(len(table), block.shape[1])}"
                        )
                else:
                    memmaps[name] = np.lib.format.open_memmap(
                        path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(table), block.shape[1]),
                    )
            memmaps[name][start : start + len(batch)] = block
        if (start // args.batch_size) % 25 == 0 and start + len(batch) < len(table):
            for m in memmaps.values():
                m.flush()
            atomic_json(progress_path, {"done": start + len(batch)})
            print(f"  {start + len(batch)}/{len(table)}", flush=True)

    for m in memmaps.values():
        m.flush()
    meta["width"] = int(next(iter(memmaps.values())).shape[1])
    if args.backend == "ntv3" and embedder.representation != meta["representation"]:
        raise SystemExit(
            f"resolved layer {embedder.representation} does not match "
            f"the recorded {meta['representation']}"
        )
    atomic_json(meta_path, meta)
    table.drop(columns=["seq"]).to_csv(out / "sequences.csv", index=False)
    atomic_json(progress_path, {"done": len(table)})
    print(f"wrote {out}/X_mean.npy and X_last.npy, width {meta['width']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", required=True, choices=("evo2", "ntv3"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--quartets", default="data/quartets.csv.gz")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="evo2_7b_base, or an InstaDeepAI/NTv3_* repo id",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="evo2: a submodule name such as blocks.26.mlp.l3. "
        "ntv3: a transformer block index, 5 for 100M, 11 for 650M",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--weights", default=None, help="evo2 local weight file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="also re-embed the 2,595 reference sequences, making the "
        "directory self-contained instead of reusing the committed ones",
    )
    parser.add_argument(
        "--include-double",
        action="store_true",
        help="also embed the AB sequences, for a later interaction probe",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = "evo2_7b_base" if args.backend == "evo2" else "InstaDeepAI/NTv3_100M_pre"
    if args.layer is None:
        args.layer = "blocks.26.mlp.l3" if args.backend == "evo2" else "5"
    run(args)


if __name__ == "__main__":
    main()
