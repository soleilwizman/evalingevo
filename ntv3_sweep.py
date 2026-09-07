#!/usr/bin/env python3
"""Sweep every NTv3 layer for element-level decodability, in one GPU pass.

Self-contained: does its own all-layer embed, then fits each layer on CPU.
Reuses evo_probe's element table, folds and ridge fit, so every number here is
comparable with the Evo 2 probe.

    python3 ntv3_sweep.py --checkpoint InstaDeepAI/NTv3_650M_pre --out results/ntv3_650m_sweep
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_probe import elements, kmers, out_of_fold

MULTIPLE = 128


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

    chunks = {}
    for start in range(0, len(seqs), args.batch_size):
        batch = seqs[start:start + args.batch_size]
        ids = torch.tensor([tok(pad(s), add_special_tokens=True)["input_ids"] for s in batch],
                           dtype=torch.long, device=device)
        got = "".join(tok.convert_ids_to_tokens(ids[0, span].tolist()))
        if got != batch[0]:
            raise SystemExit(f"token alignment wrong: {got[:20]} vs {batch[0][:20]}")
        with torch.inference_mode():
            states = model(input_ids=ids, output_hidden_states=True).hidden_states
        for i, state in enumerate(states):
            real = state[:, span].float()
            vec = real.mean(1) if args.pooling == "mean" else real[:, -1]
            chunks.setdefault(i, []).append(vec.cpu().numpy())
        if start % (args.batch_size * 25) == 0:
            print(f"  {start}/{len(seqs)}", flush=True)

    layers = sorted(chunks)
    for i in layers:
        np.save(out / f"X_{args.pooling}_L{i}.npy", np.concatenate(chunks[i]))
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "pooling": args.pooling, "layers": layers,
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
