import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRED = "results/evo2_7b_base/predictions.csv"
AUDIT = "data/audit.csv.gz"
ALPHAS = np.logspace(-2, 6, 25)
POOLINGS = ("mean", "last")


# --------------------------------------------------------------------------
# the element table: one row per distinct reference 200-mer, with its activity
# --------------------------------------------------------------------------
def elements(pred=PRED, audit=AUDIT):
    p = pd.read_csv(pred)
    a = pd.read_csv(audit)
    a["key"] = (a.v1.astype(str) + ";" + a.v2.astype(str) + ";"
                + a.center_variant.astype(str) + ";" + a.window.astype(str)
                + ";" + a.library.astype(str))
    p["key"] = p.pair_id.str.split("|").str[0]
    m = p.merge(a[["key", "refref_Log2FC", "refref_active"]], on="key", how="inner")
    if len(m) != len(p):
        raise ValueError(f"audit join dropped rows: {len(m)} of {len(p)}")

    # A reference sequence must not straddle two cross-validation groups, or
    # near-identical sequence lands on both sides of a fold boundary.
    spread = m.groupby("id_wt").group_id.nunique()
    if (spread > 1).any():
        raise ValueError(f"{(spread > 1).sum()} reference sequences span "
                         "multiple region groups; grouping is unsafe")

    el = (m.groupby("id_wt")
            .agg(seq=("seq_wt", "first"), s_wt=("s_wt", "first"),
                 activity=("refref_Log2FC", "mean"),
                 active=("refref_active", "max"), group=("group_id", "first"))
            .dropna(subset=["activity"]).reset_index()
            .rename(columns={"id_wt": "sequence_id"}))
    if el.seq.str.len().nunique() != 1:
        raise ValueError("reference sequences are not all the same length")
    if el.seq.duplicated().any():
        raise ValueError("two sequence ids share a sequence")
    return el


# --------------------------------------------------------------------------
# step 1 -- GPU. Save two summaries of one hidden layer per sequence.
# --------------------------------------------------------------------------
def embed(out, layer="blocks.26.mlp.l3", checkpoint="evo2_7b_base",
          weights=None, batch_size=4, pred=PRED, audit=AUDIT):
    import torch
    from evo2 import Evo2

    if not torch.cuda.is_available():
        raise RuntimeError("Embedding requires a supported NVIDIA GPU")

    el = elements(pred, audit)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(el)} distinct reference sequences, layer {layer}")

    model = Evo2(checkpoint, local_path=weights)
    model.model.eval()
    tokenizer = model.tokenizer

    try:
        model.model.get_submodule(layer)
    except AttributeError:
        names = sorted({n for n, _ in model.model.named_modules()
                        if "blocks." in n and n.count(".") <= 3})
        raise SystemExit(
            f"No submodule '{layer}'. Candidates in this checkpoint:\n  "
            + "\n  ".join(names[:40])
            + "\n\nPick one with --layer. Layer 26 is a medium-range "
              "convolution block; layer 27 is a long-range one.")

    pooled = {p: [] for p in POOLINGS}
    for start in range(0, len(el), batch_size):
        seqs = el.seq.iloc[start:start + batch_size].tolist()
        toks = [tokenizer.tokenize(s) for s in seqs]
        if any(len(t) != len(s) for t, s in zip(toks, seqs)):
            raise ValueError("tokenizer is not one token per nucleotide")
        ids = torch.tensor([[tokenizer.eod_id] + t for t in toks],
                           dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            output = model(ids, return_embeddings=True, layer_names=[layer])
        emb = output[1] if isinstance(output, (tuple, list)) else output
        h = emb[layer] if isinstance(emb, dict) else emb
        if h.ndim != 3 or h.shape[:2] != ids.shape:
            raise ValueError(f"unexpected embedding shape {tuple(h.shape)} "
                             f"for input {tuple(ids.shape)}")
        h = h[:, 1:, :].float()              # drop the EOD position
        pooled["mean"].append(h.mean(1).cpu().numpy())
        pooled["last"].append(h[:, -1, :].cpu().numpy())   # causal: sees all
        if start % (batch_size * 50) == 0:
            print(f"  {start}/{len(el)}", flush=True)

    for name, chunks in pooled.items():
        X = np.concatenate(chunks).astype(np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"nonfinite values in the {name}-pooled embedding")
        np.save(out / f"X_{name}.npy", X)
    el.drop(columns=["seq"]).to_csv(out / "elements.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"layer": layer, "checkpoint": checkpoint, "poolings": list(POOLINGS),
         "n": len(el), "width": int(X.shape[1])}, indent=2) + "\n")
    print(f"wrote {out}/X_mean.npy and X_last.npy, width {X.shape[1]}")


# --------------------------------------------------------------------------
# step 2 -- no GPU.
# --------------------------------------------------------------------------
def kmers(seqs, k_max=3):
    """Counts of every DNA word up to k_max. Occurrences OVERLAP, so AAAA
    contains three AAs; str.count would say two."""
    words, cols = [""], []
    for _ in range(k_max):
        words = [w + b for w in words for b in "ACGT"]
        cols += words
    out = np.zeros((len(seqs), len(cols)))
    for i, s in enumerate(seqs):
        for j, w in enumerate(cols):
            k = len(w)
            out[i, j] = sum(s[t:t + k] == w for t in range(len(s) - k + 1))
    return out


def out_of_fold(X, y, groups, folds=5):
    """Ridge with alpha chosen inside each training fold, never across it."""
    X = np.asarray(X, float)
    if not np.isfinite(X).all():
        raise ValueError("features contain NaN or infinity")
    pred = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def paired_interval(pred_a, pred_b, y, groups, n_boot=1000, seed=0):
    """Bootstrap whole groups; return the 95% interval on rho(a) - rho(b)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n_boot):
        take = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq), True)])
        diffs.append(spearmanr(pred_a[take], y[take]).statistic
                     - spearmanr(pred_b[take], y[take]).statistic)
    return np.nanpercentile(diffs, [2.5, 97.5])


def probe(embeddings, pooling="mean", pred=PRED, audit=AUDIT, folds=5,
          n_permutations=20):
    d = Path(embeddings)
    X = np.load(d / f"X_{pooling}.npy")
    el = pd.read_csv(d / "elements.csv")
    if len(X) != len(el):
        raise ValueError(f"embeddings ({len(X)}) and elements ({len(el)}) disagree")
    full = elements(pred, audit).set_index("sequence_id")
    missing = set(el.sequence_id) - set(full.index)
    if missing:
        raise ValueError(f"{len(missing)} embedded sequences are no longer in "
                         "the element table; re-run embed")
    seqs = full.seq.loc[el.sequence_id].values
    y, g = el.activity.values, el.group.values
    meta = json.loads((d / "meta.json").read_text())
    print(f"n={len(el)}  layer {meta['layer']}  pooling {pooling}  "
          f"width {X.shape[1]}\n")

    gc = np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs])
    km = kmers(seqs)

    preds, rows = {}, []
    for name, feat in [("Evo score (1 feature)", el[["s_wt"]].values),
                       ("GC content (1 feature)", gc),
                       ("DNA word counts (84 features)", km),
                       ("Evo hidden layer (probe)", X)]:
        preds[name] = out_of_fold(feat, y, g, folds)
        rows.append((name, spearmanr(preds[name], y).statistic,
                     float(np.sqrt(np.mean((preds[name] - y) ** 2)))))
    rows.append(("predict the mean", 0.0,
                 float(np.sqrt(np.mean((y - y.mean()) ** 2)))))

    w = max(len(r[0]) for r in rows)
    print(f"{'':{w}}   Spearman     RMSE")
    for name, rho, rmse in rows:
        print(f"{name:{w}}   {rho:+.4f}   {rmse:.4f}")

    # How big does a number have to be before it means anything here?
    rng = np.random.default_rng(0)
    null = [spearmanr(out_of_fold(km, rng.permutation(y), g, folds), y).statistic
            for _ in range(n_permutations)]
    print(f"\nnoise floor from {n_permutations} label permutations: "
          f"{np.mean(null):+.4f} +/- {np.std(null):.4f}")

    a, b = "Evo probe (blocks.26.mlp.l3 layer)", "DNA word counts (84 features)"
    lo, hi = paired_interval(preds[a], preds[b], y, g)
    got = dict((r[0], r[1]) for r in rows)
    print(f"\nprobe minus word counts: {got[a] - got[b]:+.4f}  "
          f"95% interval [{lo:+.4f}, {hi:+.4f}]")
    print("The information is in there, and it beats word counts." if lo > 0 else
          "No evidence Evo's representations add anything over word counts.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="GPU: pool one hidden layer per sequence")
    e.add_argument("--out", default="results/embeddings")
    e.add_argument("--layer", default="blocks.26.mlp.l3")
    e.add_argument("--checkpoint", default="evo2_7b_base")
    e.add_argument("--weights", default=None)
    e.add_argument("--batch-size", type=int, default=4)

    p = sub.add_parser("probe", help="no GPU: out-of-fold ridge against baselines")
    p.add_argument("--embeddings", default="results/embeddings")
    p.add_argument("--pooling", default="mean", choices=POOLINGS)
    p.add_argument("--folds", type=int, default=5)

    args = {k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()}
    cmd = args.pop("cmd")
    (embed if cmd == "embed" else probe)(**args)


if __name__ == "__main__":
    main()
