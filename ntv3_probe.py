import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evo_probe import elements, kmers, out_of_fold, paired_interval, PRED, AUDIT
from scipy.stats import spearmanr

MULTIPLE = 128
DEFAULT_CHECKPOINT = "InstaDeepAI/NTv3_100M_pre"


def pad_to_multiple(sequence, multiple=MULTIPLE):
    """Symmetric N padding. Returns the padded string and the left offset."""
    target = -(-len(sequence) // multiple) * multiple
    extra = target - len(sequence)
    left = extra // 2
    return "N" * left + sequence + "N" * (extra - left), left


def token_offset(tokenizer, multiple=MULTIPLE):
    """Where sequence tokens start, so pooling covers the real bases only."""
    probe_seq = "ACGT" * (multiple // 4)
    ids = tokenizer(probe_seq, add_special_tokens=True)["input_ids"]
    wanted = tokenizer.convert_tokens_to_ids(list("ACGT"))
    for start in range(len(ids) - len(probe_seq) + 1):
        if list(ids[start:start + 4]) == list(wanted) and len(ids) - start >= len(probe_seq):
            return start
    raise ValueError(f"cannot align tokens to bases: {len(ids)} ids for {len(probe_seq)} bases")


def block_hidden_states(output):
    """Read the hidden-state tensor returned by an NTv3 transformer block."""
    if hasattr(output, "hidden_states"):
        output = output.hidden_states
    elif hasattr(output, "__contains__") and "hidden_states" in output:
        output = output["hidden_states"]
    if hasattr(output, "ndim") and output.ndim == 3:
        return output
    raise ValueError("NTv3 transformer block did not return a 3D hidden_states tensor")


def embed(out, layer=11, checkpoint=DEFAULT_CHECKPOINT, revision="main",
          pred=PRED, audit=AUDIT, batch_size=8):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    el = elements(pred, audit)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    # Cast after loading: transformers 5 ignores torch_dtype, and NTv3's own code
    # builds float32 activations, so bfloat16 weights raise a dtype mismatch.
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs).float()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    if not hasattr(model, "core") or not hasattr(model.core, "transformer_blocks"):
        raise ValueError("checkpoint does not expose core.transformer_blocks")
    blocks = model.core.transformer_blocks
    if not 0 <= layer < len(blocks):
        raise ValueError(f"layer must be between 0 and {len(blocks) - 1}")
    block = blocks[layer]
    captured = {}

    def capture(_module, _inputs, output):
        captured["hidden_states"] = block_hidden_states(output)

    hook = block.register_forward_hook(capture)

    sequences = el.seq.tolist()
    length = len(sequences[0])
    padded, left = pad_to_multiple(sequences[0])
    print(f"{len(sequences)} elements, {length} bp padded to {len(padded)}, "
            f"pooling core.transformer_blocks.{layer} on {device}")

    mean_rows, last_rows = [], []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        ids = torch.tensor([tokenizer(pad_to_multiple(s)[0],
                                      add_special_tokens=True)["input_ids"] for s in batch],
                           dtype=torch.long, device=device)
        with torch.inference_mode():
            model(input_ids=ids)
        hidden = captured.pop("hidden_states")
        real = hidden.float()
        if not torch.isfinite(real).all():
            raise ValueError(f"core.transformer_blocks.{layer} produced non-finite activations")
        mean_rows.append(real.mean(1).cpu().numpy())
        last_rows.append(real[:, -1].cpu().numpy())
        if start % (batch_size * 25) == 0:
            print(f"  {start}/{len(sequences)}", flush=True)

    hook.remove()
    matrices = {"mean": np.concatenate(mean_rows), "last": np.concatenate(last_rows)}
    for name, matrix in matrices.items():
        np.save(out / f"X_{name}.npy", matrix)
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
         {"model": "ntv3", "checkpoint": checkpoint, "revision": revision,
         "layer": f"core.transformer_blocks.{layer}", "width": int(matrices["mean"].shape[1]),
         "n": int(len(el)), "padded_length": len(padded)}, indent=2) + "\n")
    print(f"wrote {out}/X_mean.npy and X_last.npy, width {matrices['mean'].shape[1]}")


def probe(embeddings, pooling="mean", pred=PRED, audit=AUDIT, folds=5, n_permutations=20):
    d = Path(embeddings)
    X = np.load(d / f"X_{pooling}.npy")
    el = pd.read_csv(d / "elements.csv")
    if len(X) != len(el):
        raise ValueError(f"embeddings ({len(X)}) and elements ({len(el)}) disagree")
    full = elements(pred, audit).set_index("sequence_id")
    missing = set(el.sequence_id) - set(full.index)
    if missing:
        raise ValueError(f"{len(missing)} embedded sequences are no longer in the "
                         "element table; re-run embed")
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values
    meta = json.loads((d / "meta.json").read_text())
    print(f"n={len(el)}  {meta['checkpoint']}  layer {meta['layer']}  "
          f"pooling {pooling}  width {X.shape[1]}\n")

    gc = np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs])
    km = kmers(seqs)

    preds, rows = {}, []
    for name, feat in [("GC content (1 feature)", gc),
                       ("DNA word counts (84 features)", km),
                       ("NTv3 hidden layer (probe)", X),
                       ("NTv3 hidden layer + word counts", np.hstack([X, km]))]:
        preds[name] = out_of_fold(feat, y, g, folds)
        rows.append((name, spearmanr(preds[name], y).statistic,
                     float(np.sqrt(np.mean((preds[name] - y) ** 2)))))
    rows.append(("predict the mean", 0.0, float(np.sqrt(np.mean((y - y.mean()) ** 2)))))

    width = max(len(r[0]) for r in rows)
    print(f"{'':{width}}   Spearman     RMSE")
    for name, rho, rmse in rows:
        print(f"{name:{width}}   {rho:+.4f}   {rmse:.4f}")

    rng = np.random.default_rng(0)
    shuffled = [spearmanr(out_of_fold(X, y[rng.permutation(len(y))], g, folds), y).statistic
                for _ in range(n_permutations)]
    print(f"\nnoise floor from {n_permutations} label permutations: "
          f"{np.mean(shuffled):+.4f} +/- {np.std(shuffled):.4f}")

    low, high = paired_interval(preds["NTv3 hidden layer (probe)"],
                                preds["DNA word counts (84 features)"], y, g)
    margin = (spearmanr(preds["NTv3 hidden layer (probe)"], y).statistic
              - spearmanr(preds["DNA word counts (84 features)"], y).statistic)
    print(f"probe minus word counts: {margin:+.4f}  95% interval [{low:+.4f}, {high:+.4f}]")
    print("The information is in there, and it beats word counts."
          if low > 0 else
          "Not distinguishable from word counts on this evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="GPU. Pool one hidden layer per element.")
    e.add_argument("--out", default="results/ntv3_probe")
    e.add_argument("--layer", type=int, default=11,
                   help="actual NTv3 transformer block index (default 11)")
    e.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    e.add_argument("--revision", default="main")
    e.add_argument("--batch-size", type=int, default=8)

    p = sub.add_parser("probe", help="CPU. Fit ridge on the saved embeddings.")
    p.add_argument("--embeddings", default="results/ntv3_probe")
    p.add_argument("--pooling", choices=("mean", "last"), default="mean")
    p.add_argument("--folds", type=int, default=5)

    args = vars(parser.parse_args())
    cmd = args.pop("cmd")
    (embed if cmd == "embed" else probe)(**args)


if __name__ == "__main__":
    main()
