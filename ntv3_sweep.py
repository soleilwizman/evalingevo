#!/usr/bin/env python3
"""Sweep every NTv3 layer for element-level decodability, in one GPU pass.

Self-contained: does its own all-layer embed, then fits each layer on CPU.
Reuses evo_probe's element table, folds and ridge fit, so every number here is
comparable with the Evo 2 probe.

    python3 ntv3_sweep.py --checkpoint InstaDeepAI/NTv3_650M_pre --out results/ntv3_650m_sweep
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_probe import elements, kmers, out_of_fold

MULTIPLE = 128


def find_layers(model):
    """Find the repeated user-facing transformer layers by module name."""
    patterns = (
        re.compile(r"^(.*(?:^|\.)(?:transformer_blocks|layers|blocks|h))\.(\d+)$"),
        re.compile(r"^(.*\.encoder\.layer)\.(\d+)$"),
    )
    candidates = {}
    for name, module in model.named_modules():
        for pattern in patterns:
            match = pattern.match(name)
            if match:
                candidates.setdefault(match.group(1), {})[int(match.group(2))] = (name, module)
                break

    complete = []
    for parent, items in candidates.items():
        indices = sorted(items)
        if indices == list(range(len(indices))) and len(indices) > 1:
            complete.append((len(indices), parent, items))
    if not complete:
        raise RuntimeError(
            "Could not find a contiguous repeated layer stack. Available module names:\n"
            + "\n".join(name for name, _ in model.named_modules() if name)[:4000]
        )

    _, parent, items = max(complete, key=lambda entry: entry[0])
    layers = [items[i] for i in range(len(items))]
    print(f"using {len(layers)} named layers under {parent}", flush=True)
    for index, (name, _) in enumerate(layers):
        print(f"  L{index}: {name}", flush=True)
    return layers


def tensor_output(output, name):
    """Extract a batch-first hidden-state tensor from a module output."""
    tensors = []

    def visit(value):
        if hasattr(value, "ndim") and value.ndim == 3:
            tensors.append(value)
        elif hasattr(value, "items"):
            for _, item in value.items():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, "_fields"):
            for field in value._fields:
                visit(getattr(value, field))
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item)

    visit(output)
    if not tensors:
        raise RuntimeError(f"layer {name} returned an unexpected output")
    return tensors[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_pre")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default="results/ntv3_sweep")
    ap.add_argument("--pooling", choices=("mean", "last"), default="mean")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    el = elements()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kw = {"trust_remote_code": True, "revision": args.revision}
    tok = AutoTokenizer.from_pretrained(args.checkpoint, **kw)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint, **kw).float()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    named_layers = find_layers(model)

    # where sequence tokens start, so pooling covers the real bases only
    probe_seq = "ACGT" * (MULTIPLE // 4)
    ids0 = tok(probe_seq, add_special_tokens=True)["input_ids"]
    want = tok.convert_tokens_to_ids(list("ACGT"))
    offset = next(i for i in range(len(ids0) - len(probe_seq) + 1)
                  if list(ids0[i:i + 4]) == list(want)
                  and len(ids0) - i >= len(probe_seq))

    seqs = el.seq.tolist()
    length = len(seqs[0])
    target = -(-length // MULTIPLE) * MULTIPLE
    left = (target - length) // 2
    span = slice(offset + left, offset + left + length)

    def pad(s):
        return "N" * left + s + "N" * (target - length - left)

    print(f"{len(seqs)} elements, {length} bp padded to {target}, "
          f"pooling {span.start}:{span.stop} on {device}", flush=True)

    chunks = {i: [] for i in range(len(named_layers))}
    captured = {}

    def capture(index, name):
        def hook(_module, _inputs, output):
            captured[index] = tensor_output(output, name)
        return hook

    hooks = [module.register_forward_hook(capture(i, name))
             for i, (name, module) in enumerate(named_layers)]
    for start in range(0, len(seqs), args.batch_size):
        batch = seqs[start:start + args.batch_size]
        ids = torch.tensor([tok(pad(s), add_special_tokens=True)["input_ids"] for s in batch],
                           dtype=torch.long, device=device)
        got = "".join(tok.convert_ids_to_tokens(ids[0, span].tolist()))
        if got != batch[0]:
            raise SystemExit(f"token alignment wrong: {got[:20]} vs {batch[0][:20]}")
        captured.clear()
        with torch.inference_mode():
            model(input_ids=ids)
        if len(captured) != len(named_layers):
            missing = sorted(set(range(len(named_layers))) - set(captured))
            raise RuntimeError(f"named layer hooks did not fire: {missing}")
        for i, (name, _) in enumerate(named_layers):
            state = captured[i]
            if state.shape[0] != len(batch):
                raise RuntimeError(f"layer {name} is not batch-first: {tuple(state.shape)}")
            # Transformer blocks operate at the two-position bottleneck; the
            # convolutional stack operates at the 256-token sequence length.
            real = state[:, span] if state.shape[1] >= span.stop else state
            real = real.float()
            vec = real.mean(1) if args.pooling == "mean" else real[:, -1]
            if not torch.isfinite(vec).all():
                raise RuntimeError(f"named layer {name} produced non-finite activations")
            chunks.setdefault(i, []).append(vec.cpu().numpy())
        if start % (args.batch_size * 25) == 0:
            print(f"  {start}/{len(seqs)}", flush=True)

    for hook in hooks:
        hook.remove()
    layers = sorted(chunks)
    for i in layers:
        np.save(out / f"X_{args.pooling}_L{i}.npy", np.concatenate(chunks[i]))
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "pooling": args.pooling, "layers": layers,
         "layer_names": [name for name, _ in named_layers],
         "width": int(np.concatenate(chunks[layers[-1]]).shape[1]), "n": len(el)},
        indent=2) + "\n")

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
