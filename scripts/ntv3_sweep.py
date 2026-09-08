#!/usr/bin/env python3
"""Sweep every NTv3 layer for element-level decodability, in one GPU pass.

Self-contained: does its own all-layer embed, then fits each layer on CPU.
Reuses evo_probe's element table, folds and ridge fit, so every number here is
comparable with the Evo 2 probe.

    python3 scripts/ntv3_sweep.py --checkpoint InstaDeepAI/NTv3_650M_pre --out results/ntv3_650m_sweep
"""

import argparse
import json
from pathlib import Path

import numpy as np
from element_data import elements
from evo_probe import kmers
from scipy.stats import spearmanr
from validation import out_of_fold

MULTIPLE = 128


def find_layers(model):
    if not hasattr(model, "core") or not hasattr(model.core, "transformer_blocks"):
        raise ValueError("expected NTv3 core.transformer_blocks; refusing to guess another stack")
    return [
        (f"core.transformer_blocks.{i}.final_layer_norm", block.final_layer_norm)
        for i, block in enumerate(model.core.transformer_blocks)
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_pre")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default="results/ntv3_sweep")
    ap.add_argument("--pooling", choices=("mean", "last"), default="mean")
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="capture only this transformer layer (for example, 11)",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import torch
    from model_runtime import (
        POOLING_PROTOCOL,
        capture_layers,
        load_hf,
        pool_real_bases,
        tokenize_ntv3,
    )

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    el = elements()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, device = load_hf(args.checkpoint, args.revision)
    all_named_layers = find_layers(model)
    if args.layer is None:
        layer_indices = list(range(len(all_named_layers)))
        named_layers = all_named_layers
    else:
        if not 0 <= args.layer < len(all_named_layers):
            raise SystemExit(f"--layer must be between 0 and {len(all_named_layers) - 1}")
        layer_indices = [args.layer]
        named_layers = [all_named_layers[args.layer]]
        print(f"capturing only transformer layer {args.layer}", flush=True)

    seqs = el.seq.tolist()
    chunks = {i: [] for i in layer_indices}
    with capture_layers(named_layers) as captured:
        for start in range(0, len(seqs), args.batch_size):
            batch = seqs[start : start + args.batch_size]
            ids, left, padded_length = tokenize_ntv3(tok, batch, device)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids)
            for i, (name, _) in zip(layer_indices, named_layers):
                state = captured[name]
                if state.shape[0] != len(batch):
                    raise ValueError(f"{name} is not batch-first")
                mean, last = pool_real_bases(state, left, len(batch[0]), padded_length)
                chunks[i].append(mean if args.pooling == "mean" else last)
            if start % (args.batch_size * 25) == 0:
                print(f"  {start}/{len(seqs)}", flush=True)
    layers = sorted(chunks)
    for i in layers:
        np.save(out / f"X_{args.pooling}_L{i}.npy", np.concatenate(chunks[i]))
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "pooling": args.pooling,
                "layers": layers,
                "layer_names": [name for name, _ in named_layers],
                "selected_layer": args.layer,
                "revision": args.revision,
                "pooling_protocol": POOLING_PROTOCOL,
                "pooled_over_real_bases_only": True,
                "width": int(np.concatenate(chunks[layers[-1]]).shape[1]),
                "n": len(el),
            },
            indent=2,
        )
        + "\n"
    )

    y, g = el.activity.values, el.group.values
    base = spearmanr(out_of_fold(kmers(el.seq.values), y, g, args.folds), y).statistic
    print(f"\n{args.checkpoint}  pooling {args.pooling}  n={len(el)}")
    print(f"1-2-3 k-mer counts: {base:+.4f}\n")
    print("layer   Spearman   vs k-mers")
    rows = []
    for i in layers:
        X = np.load(out / f"X_{args.pooling}_L{i}.npy")
        rho = spearmanr(out_of_fold(X, y, g, args.folds), y).statistic
        rows.append((i, rho))
        print(f"{i:>5}   {rho:+.4f}   {rho - base:+.4f}", flush=True)
    best, rho = max(rows, key=lambda r: r[1])
    print(f"\nbest layer {best} at {rho:+.4f}, {rho - base:+.4f} against k-mers")
    print("Layer was swept, not chosen, so report the whole curve rather than the peak.")


if __name__ == "__main__":
    main()
