#!/usr/bin/env python3
"""Extract NTv3 representations from anywhere on the U, including the top of it.

Why this exists: `ntv3_probe.py embed` hooks `core.transformer_blocks.<i>`, which sits
at the bottom of the U-Net. With 7 downsamples a 200-mer padded to 256 is *2 positions*
there, so "mean pooling" averages two vectors and per-base resolution is gone. That is
fine for the 8 kb inputs the model was built for and useless for a 200 bp oligo.

The deconv tower puts the resolution back. Verified against the checkpoint's own remote
code (`InstaDeepAI/ntv3_base_model--modeling_ntv3_pretrained`), `Core.forward` appends to
`hidden_states` in exactly this order:

    conv tower          one per block, taken BEFORE its avg_pool, so 256,128,64,32,16,8,4
    transformer tower   one per layer, all at the 2-position bottleneck
    deconv tower        one per block, AFTER upsample and skip connection, so 4,8,...,256

so `hidden_states[-1]` is the post-deconv output at full input length, which is what the
model card calls "final embedding (after deconv tower)". Skip connections are on
(`use_skip_connection: true`), so that representation carries the conv tower's
high-resolution detail as well as whatever the transformer did at the bottleneck.

`filter_list` is a linspace from `conv_init_embed_dim` to `embed_dim`; both are equal in
the released checkpoints, so every stage has the same width and these matrices drop
straight into `ntv3_probe.py probe` and `layer_curve.py`.

Two caveats this script does not paper over. NTv3's forward pass builds no attention
mask, so the N padding is attended like real sequence; the pooled output therefore
excludes the pad positions by slicing, but the pads still influenced the activations.
And a 200 bp input reaches the bottleneck as 2 positions no matter which stage you read,
so this recovers resolution, not context. For context you need longer inputs.

    python3 ntv3_unet.py list  --checkpoint InstaDeepAI/NTv3_650M_pre --revision main
    python3 ntv3_unet.py embed --representation deconv_final --revision main \
        --checkpoint InstaDeepAI/NTv3_650M_pre --out results/ntv3_650m_deconv
    python3 ntv3_unet.py embed --representation all_deconv --revision main \
        --checkpoint InstaDeepAI/NTv3_650M_pre --out results/ntv3_650m_deconv_sweep
    python3 layer_curve.py results/ntv3_650m_deconv_sweep     # the whole up-slope
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evo_probe import elements, AUDIT, PRED
from ntv3_probe import MULTIPLE, pad_to_multiple

DEFAULT_CHECKPOINT = "InstaDeepAI/NTv3_650M_pre"


def load(checkpoint, revision, device):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs)
    # bfloat16 weights collide with NTv3's float32 internals under transformers 5
    model = model.float().to(device).eval()
    if not hasattr(model, "core"):
        raise SystemExit("checkpoint does not expose .core; not an NTv3 pretrained model")
    return tokenizer, model


def stage_names(model, padded_length):
    """Name every hidden state and say what length it must have.

    Derived from the module counts rather than the config, then checked against the
    tensors the model actually returns, so a change upstream fails loudly here.
    """
    core = model.core
    n_conv = len(core.conv_tower_blocks)
    n_transformer = len(core.transformer_blocks)
    n_deconv = len(core.deconv_tower_blocks)
    bottleneck = padded_length // (2 ** n_conv)
    if bottleneck < 1:
        raise SystemExit(f"{padded_length} bp cannot survive {n_conv} halvings; "
                         f"use an input of at least {2 ** n_conv} bp")

    stages = []
    for i in range(n_conv):                       # taken before each avg_pool
        stages.append((f"conv_{i + 1}", padded_length // (2 ** i)))
    for i in range(n_transformer):
        stages.append((f"transformer_{i + 1}", bottleneck))
    for i in range(n_deconv):                     # after upsample + skip
        stages.append((f"deconv_{i + 1}", bottleneck * (2 ** (i + 1))))
    return stages


def hidden_states(model, ids, padded_length):
    import torch

    with torch.inference_mode():
        out = model.core(input_ids=ids, output_hidden_states=True)
    states = out["hidden_states"]
    stages = stage_names(model, padded_length)
    if len(states) != len(stages):
        raise SystemExit(f"expected {len(stages)} hidden states, got {len(states)}; "
                         "the checkpoint's remote code has changed, re-read it")
    for (name, expected), tensor in zip(stages, states):
        if tensor.shape[1] != expected:
            raise SystemExit(f"{name}: expected {expected} positions, got {tensor.shape[1]}")
    return dict(zip([s[0] for s in stages], states)), stages


def tokenize(tokenizer, sequences, device):
    """Model card's recipe: no special tokens, N-pad to a multiple of 128."""
    import torch

    padded, lefts = zip(*[pad_to_multiple(s) for s in sequences])
    if len(set(lefts)) != 1 or len(set(map(len, padded))) != 1:
        raise SystemExit("batch mixes sequence lengths; group by length first")
    rows = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in padded]
    if any(len(r) != len(padded[0]) for r in rows):
        raise SystemExit("tokenizer is not one token per character on this batch")
    if len(padded[0]) % MULTIPLE:
        raise SystemExit(f"padded length {len(padded[0])} is not a multiple of {MULTIPLE}")
    return torch.tensor(rows, dtype=torch.long, device=device), lefts[0], len(padded[0])


def pool(tensor, left, real_length, padded_length, upsample="none"):
    """Mean and last over the real bases when the stage is at per-base resolution.

    A stage is per-base only when it has one position per input token; the deconv
    output is aligned 1:1 with the tokens, which is what lets the LM head emit
    per-base logits. Anywhere below that there is no way to say which positions are
    the real sequence, so everything is pooled and the caller is told.

    ``upsample="repeat"`` is BEND's convention for models coarser than one vector per
    base: "we repeat each embedding vector to the length of the sequence represented
    by its token" (github.com/frederikkemarin/BEND, ``upsample_embeddings=True``).
    It adds no information; it makes shapes match so one probe can run on every model.
    For a 200-mer padded to 256 the real bases split 100/100 across the two bottleneck
    positions, so repeat-then-pool equals plain pooling to float precision. It changes
    nothing for an element-level probe and matters only per position.
    """
    import torch

    if upsample == "repeat" and tensor.shape[1] < padded_length:
        if padded_length % tensor.shape[1]:
            raise SystemExit(f"cannot repeat {tensor.shape[1]} positions to {padded_length}")
        tensor = torch.repeat_interleave(tensor, padded_length // tensor.shape[1], dim=1)
    per_base = tensor.shape[1] == padded_length
    window = tensor[:, left:left + real_length, :] if per_base else tensor
    values = window.float()
    if not torch.isfinite(values).all():
        raise SystemExit("non-finite activations in the selected stage")
    return values.mean(1).cpu().numpy(), values[:, -1].cpu().numpy(), per_base


def run_list(args):
    tokenizer, model = load(args.checkpoint, args.revision, args.device)
    probe_length = 200
    ids, left, padded_length = tokenize(tokenizer, ["ACGT" * (probe_length // 4)], args.device)
    states, stages = hidden_states(model, ids, padded_length)
    print(f"{args.checkpoint}   {probe_length} bp padded to {padded_length}   "
          f"left offset {left}\n")
    print(f"{'stage':<18}{'positions':>10}{'width':>8}   per-base?")
    for name, _ in stages:
        tensor = states[name]
        flag = "yes" if tensor.shape[1] == padded_length else "no"
        print(f"{name:<18}{tensor.shape[1]:>10}{tensor.shape[2]:>8}   {flag}")
    print("\ndeconv_final is an alias for the last deconv stage.")


def run_embed(args):
    import torch

    el = elements(args.pred, args.audit)
    if args.limit:
        el = el.head(args.limit)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load(args.checkpoint, args.revision, args.device)

    sequences = el.seq.tolist()
    real_length = len(sequences[0])
    if any(len(s) != real_length for s in sequences):
        raise SystemExit("element table mixes sequence lengths")
    probe_ids, left, padded_length = tokenize(tokenizer, sequences[:1], args.device)
    _, stages = hidden_states(model, probe_ids, padded_length)
    available = [s[0] for s in stages]
    deconv = [n for n in available if n.startswith("deconv_")]

    if args.representation == "all_deconv":
        wanted = deconv
    elif args.representation == "deconv_final":
        wanted = [deconv[-1]]
    elif args.representation in available:
        wanted = [args.representation]
    else:
        raise SystemExit(f"unknown representation {args.representation!r}; "
                         f"choose from all_deconv, deconv_final, or {', '.join(available)}")

    print(f"{len(el)} elements, {real_length} bp padded to {padded_length}, "
          f"left offset {left}, stages {', '.join(wanted)}, device {args.device}")

    collected = {name: {"mean": [], "last": []} for name in wanted}
    per_base = {}
    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start:start + args.batch_size]
        ids, batch_left, batch_padded = tokenize(tokenizer, batch, args.device)
        if batch_left != left or batch_padded != padded_length:
            raise SystemExit("padding drifted between batches")
        states, _ = hidden_states(model, ids, padded_length)
        for name in wanted:
            mean, last, flag = pool(states[name], left, real_length, padded_length,
                                     args.upsample)
            collected[name]["mean"].append(mean)
            collected[name]["last"].append(last)
            per_base[name] = flag
        if start % (args.batch_size * 25) == 0:
            print(f"  {start}/{len(sequences)}", flush=True)

    sweep = len(wanted) > 1
    widths = {}
    for name in wanted:
        suffix = f"_L{name.split('_')[1]}" if sweep else ""
        for pooling in ("mean", "last"):
            matrix = np.concatenate(collected[name][pooling])
            np.save(out / f"X_{pooling}{suffix}.npy", matrix)
            if pooling == "mean":
                widths[name] = int(matrix.shape[1])

    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    label = wanted[0] if not sweep else f"{wanted[0]}..{wanted[-1]}"
    (out / "meta.json").write_text(json.dumps(
        {"model": "ntv3", "checkpoint": args.checkpoint, "revision": args.revision,
         "layer": f"{label} (U-Net deconv tower)" if label.startswith("deconv")
                  else f"{label} (bottleneck)",
         "representations": wanted, "width": widths[wanted[0]],
         "n": int(len(el)), "padded_length": padded_length, "left_offset": left,
         "real_length": real_length,
         "pooled_over_real_bases_only": {n: per_base[n] for n in wanted},
         "upsample": args.upsample,
         "note": "NTv3 builds no attention mask, so N padding is attended even though "
                 "pooling excludes those positions"}, indent=2) + "\n")
    for name in wanted:
        scope = "real bases only" if per_base[name] else "all positions (below per-base)"
        print(f"  {name}: width {widths[name]}, pooled over {scope}")
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = lambda p: (
        p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT),
        p.add_argument("--revision", default=None,
                       help="required for a real run; pin the checkpoint commit"),
        p.add_argument("--device", default="cuda:0"))

    listing = sub.add_parser("list", help="print every stage of the U and its resolution")
    common(listing)
    listing.set_defaults(upsample="none")

    embed = sub.add_parser("embed", help="pool one stage (or the whole deconv tower)")
    common(embed)
    embed.add_argument("--representation", default="deconv_final",
                       help="deconv_final, all_deconv, deconv_<k>, transformer_<k>, conv_<k>")
    embed.add_argument("--out", default="results/ntv3_deconv")
    embed.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    embed.add_argument("--upsample", default="none", choices=("none", "repeat"),
                       help="repeat = BEND's convention, repeat each vector to the span "
                            "its token covers; a no-op for pooled element embeddings")
    embed.add_argument("--limit", type=int, default=0, help="smoke test on the first N elements")
    embed.add_argument("--pred", default=PRED)
    embed.add_argument("--audit", default=AUDIT)

    args = parser.parse_args()
    if args.cmd == "embed" and not args.revision:
        raise SystemExit("--revision is mandatory for an embedding run")
    (run_list if args.cmd == "list" else run_embed)(args)


if __name__ == "__main__":
    main()
