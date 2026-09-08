#!/usr/bin/env python3
"""Fit the NTv3 variant probes under single_variant_spearman.py's protocol.

The committed variant embeddings hold the alternate sequences only
(include_reference: false), so the probe feature is h(alt) - h(ref), with the
reference vector taken from the element embeddings that already exist.

Every readout is refit on whichever variants have both vectors, so the probe
and the baselines it is compared against are scored on the same rows.

    python3 variant_probe_fit.py --variants results/ntv3_650m_variants \
        --references results/ntv3_650m_final --label "NTv3 650M probe"
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_epistasis import gc_fraction, group_boot, out_of_fold_linear
from evo_probe import kmers
from single_variant import single_variants


def sha(seq):
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    var, ref = Path(a.variants), Path(a.references)
    Xv = np.load(var / f"X_{a.pooling}.npy")
    sv = pd.read_csv(var / "sequences.csv")
    if len(sv) != len(Xv):
        raise SystemExit(f"{var}: {len(sv)} rows but {len(Xv)} vectors")
    Xr = np.load(ref / f"X_{a.pooling}.npy")
    sr = pd.read_csv(ref / "elements.csv")
    if len(sr) != len(Xr):
        raise SystemExit(f"{ref}: {len(sr)} rows but {len(Xr)} vectors")
    if Xv.shape[1] != Xr.shape[1]:
        raise SystemExit(f"width mismatch: variants {Xv.shape[1]} vs references {Xr.shape[1]}")
    vi = {s: i for i, s in enumerate(sv.sequence_id)}
    ri = {s: i for i, s in enumerate(sr.sequence_id)}

    table = single_variants("results/evo2_7b_base/predictions.csv")
    table = table.assign(ref_id=[sha(s) for s in table.seq_ref])
    have = np.array([(v in vi) and (r in ri)
                     for v, r in zip(table.sequence_id, table.ref_id)])
    kept = table[have].reset_index(drop=True)
    print(f"{a.label}: {have.sum()} of {len(table)} variants have both vectors")
    if not have.any():
        raise SystemExit("no overlap between the variant and reference tables")

    diff = np.stack([Xv[vi[v]] - Xr[ri[r]]
                     for v, r in zip(kept.sequence_id, kept.ref_id)])
    refonly = np.stack([Xr[ri[r]] for r in kept.ref_id])
    y, g = kept.y.to_numpy(), kept.group_id.to_numpy()
    seqs, refs = kept.seq.tolist(), kept.seq_ref.tolist()

    rows = [
        (f"{a.label} (h(alt) - h(ref))", diff, "probe"),
        (f"{a.label}, reference only (control)", refonly, "probe"),
        ("1/2/3-mer delta", kmers(seqs) - kmers(refs), "baseline"),
        ("GC content", gc_fraction(seqs).reshape(-1, 1), "baseline"),
    ]
    # No likelihood row here on purpose: delta_score in this table is Evo 2's,
    # taken from the shared variant table, not the checkpoint being probed.
    out = {"label": a.label, "n": int(have.sum()), "n_total": int(len(table)),
           "pooling": a.pooling, "folds": a.folds, "seed": a.seed,
           "variants": str(var), "references": str(ref),
           "width": int(Xv.shape[1]), "readouts": []}
    for name, feat, kind in rows:
        pred = out_of_fold_linear(feat, y, g, a.folds,
                                  ridge=feat.shape[1] > 1, seed=a.seed)
        rho = float(spearmanr(pred, y).statistic)
        lo, hi = group_boot(g, lambda i: spearmanr(pred[i], y[i]).statistic,
                            a.bootstrap, a.seed)
        out["readouts"].append({"label": name, "kind": kind, "spearman": rho,
                                "ci": [float(lo), float(hi)],
                                "features": int(feat.shape[1])})
        print(f"  {name:44} {rho:+.4f}  [{lo:+.4f}, {hi:+.4f}]", flush=True)

    path = Path(a.out) if a.out else var / "variant_probe.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
