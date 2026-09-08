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
from scipy.stats import spearmanr

from evo_probe import elements, kmers, out_of_fold, paired_interval

MULTIPLE = 128


def layer(embeddings, layer, pooling="mean", folds=5):
    d = Path(embeddings)
    path = d / f"X_{pooling}_L{layer}.npy"
    if not path.exists():
        path = d / f"X_{pooling}.npy"
        if not path.exists():
            found = sorted(int(f.stem.split("_L")[1]) for f in d.glob(f"X_{pooling}_L*.npy"))
            raise SystemExit(f"no matrix for layer {layer} in {d}. Available: {found}")
    X = np.load(path)
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X).all(1)).sum())
        raise SystemExit(f"layer {layer} has {bad} non-finite rows; it cannot be fitted")

    el = pd.read_csv(d / "elements.csv")
    full = elements().set_index("sequence_id")
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values

    gc = np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs])
    km = kmers(seqs)
    meta = json.loads((d / "meta.json").read_text())
    print(f"{meta.get('checkpoint', d)}   layer {layer}   pooling {pooling}   "
          f"n={len(el)}   width {X.shape[1]}\n")

    preds, rows = {}, []
    for name, feat in [("GC content (1 feature)", gc),
                       ("1-2-3 k-mer counts (84 features)", km),
                       (f"NTv3 layer {layer} (probe)", X),
                       (f"NTv3 layer {layer} + k-mers", np.hstack([X, km]))]:
        preds[name] = out_of_fold(feat, y, g, folds)
        rows.append((name, spearmanr(preds[name], y).statistic,
                     float(np.sqrt(np.mean((preds[name] - y) ** 2)))))
    rows.append(("predict the mean", 0.0, float(np.sqrt(np.mean((y - y.mean()) ** 2)))))

    width = max(len(r[0]) for r in rows)
    print(f"{'':{width}}   Spearman     RMSE")
    for name, rho, rmse in rows:
        print(f"{name:{width}}   {rho:+.4f}   {rmse:.4f}")

    a, b = f"NTv3 layer {layer} (probe)", "1-2-3 k-mer counts (84 features)"
    margin = spearmanr(preds[a], y).statistic - spearmanr(preds[b], y).statistic
    low, high = paired_interval(preds[a], preds[b], y, g)
    print(f"\nprobe minus k-mers: {margin:+.4f}  95% interval [{low:+.4f}, {high:+.4f}]")
    print("Beats k-mer counts." if low > 0 else
          "Reliably WORSE than k-mer counts." if high < 0 else
          "Not distinguishable from k-mer counts on this evidence.")


def likelihood(checkpoint="InstaDeepAI/NTv3_650M_pre", revision="main",
               out="results/ntv3_650m_elements", batch_size=100):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    el = elements()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    kw = {"trust_remote_code": True, "revision": revision}
    tok = AutoTokenizer.from_pretrained(checkpoint, **kw)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kw).float()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise SystemExit("tokenizer exposes no mask token")

    probe_seq = "ACGT" * (MULTIPLE // 4)
    ids0 = tok(probe_seq, add_special_tokens=True)["input_ids"]
    want = tok.convert_tokens_to_ids(list("ACGT"))
    offset = next(i for i in range(len(ids0) - len(probe_seq) + 1)
                  if list(ids0[i:i + 4]) == list(want) and len(ids0) - i >= len(probe_seq))

    seqs = el.seq.tolist()
    length = len(seqs[0])
    target = -(-length // MULTIPLE) * MULTIPLE
    left = (target - length) // 2
    positions = [offset + left + i for i in range(length)]

    def complement(s):
        return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]

    def score(sequence):
        padded = "N" * left + sequence + "N" * (target - length - left)
        ids = torch.tensor(tok(padded, add_special_tokens=True)["input_ids"],
                           dtype=torch.long, device=device)
        truth = ids[positions].clone()
        got = "".join(tok.convert_ids_to_tokens(truth.tolist()))
        if got != sequence:
            raise SystemExit(f"token alignment wrong: {got[:20]} vs {sequence[:20]}")
        total = 0.0
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            batch = ids.unsqueeze(0).repeat(len(chunk), 1)
            index = torch.tensor(chunk, device=device)
            batch[torch.arange(len(chunk), device=device), index] = mask_id
            with torch.inference_mode():
                logits = model(input_ids=batch).logits
            logp = torch.log_softmax(logits.float(), -1)
            picked = logp[torch.arange(len(chunk), device=device), index,
                          truth[start:start + len(chunk)]]
            total += float(picked.detach().cpu().double().sum())
        return total

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
    print(f"  active elements only {spearmanr(active.likelihood, active.activity).statistic:+.4f}"
          f"   (n={len(active)})")
    print(f"  per base            {np.mean(scores) / length:+.4f}")
    print(f"\nwrote {out}/element_likelihood.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
