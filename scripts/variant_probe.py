#!/usr/bin/env python3
"""Variant probe across model families, on one protocol.

    # GPU, once per model
    python3 scripts/variant_probe.py embed --model evo2     --out results/vp_evo2
    python3 scripts/variant_probe.py embed --model ntv3     --out results/vp_ntv3_100m \
        --checkpoint InstaDeepAI/NTv3_100M_pre
    python3 scripts/variant_probe.py embed --model ntv3     --out results/vp_ntv3_650m \
        --checkpoint InstaDeepAI/NTv3_650M_pre
    python3 scripts/variant_probe.py embed --model dnabert2 --out results/vp_dnabert2

    # CPU
    python3 scripts/variant_probe.py probe   --embeddings results/vp_evo2
    python3 scripts/variant_probe.py compare --embeddings results/vp_evo2 results/vp_ntv3_100m \
        results/vp_ntv3_650m results/vp_dnabert2

Each model is read in its OWN units, not forced into a shared one.

  Evo 2 and NTv3 read one base at a time, so a sequence is 200 vectors and the
  variant has a vector of its own.
  DNABERT-2 uses BPE, so a sequence is about 40 vectors, each covering four or
  five bases, and the variant has no vector of its own. It gets the vector of
  the token containing it.

Sequence pooling is a plain mean over whatever those units are, which is what
each model's own documentation recommends. An earlier version broadcast
DNABERT-2's token vectors out to bases before pooling, which is a
length-weighted mean over tokens and not a representation the model produces.

The consequence to keep in mind when reading the results: the variant-level
feature is a single base for Evo 2 and NTv3 and a four-to-five base chunk for
DNABERT-2. That is each model at its native resolution, which is the fair
comparison, but it is not the same quantity.

Three feature sets per model, and the third decides what a win means:

  d_mean   mutant mean-pooled minus reference mean-pooled, averaged over 200
           positions, so one changed base is heavily diluted
  d_pos    the same difference read only at the base that changed
  wt_mean  the reference's own embedding, which knows nothing about which base
           changed. Some elements are more fragile than others, so a probe can
           score by recognising a sensitive element. If this matches the other
           two, that is what happened.

GC is not a separate row. For a single substitution the GC change is +1, -1 or
0, which is contained in the first four columns of the k-mer difference
baseline, so it is covered rather than omitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from artifact_io import source_hashes
from benchmark_data import file_hash
from model_adapters import DNABERT2Adapter as DNABERT2Adapter
from model_adapters import Evo2Adapter as Evo2Adapter
from model_adapters import NTv3Adapter as NTv3Adapter
from model_runtime import POOLING_PROTOCOL
from validation import out_of_fold as out_of_fold
from variant_data import noise_ceiling as noise_ceiling
from variant_data import observations as observations
from variant_data import within_element_accuracy as within_element_accuracy
from variant_evaluation import build_features as build_features
from variant_evaluation import compare as compare
from variant_evaluation import probe as probe
from variant_evaluation import score_one as score_one

AUDIT = "data/audit.csv.gz"

QUARTETS = "data/quartets.csv.gz"
NTV3_MULTIPLE = 128
SEED = 0


# ------------------------------------------------------------------- adapters


ADAPTERS = {"evo2": Evo2Adapter, "ntv3": NTv3Adapter, "dnabert2": DNABERT2Adapter}
DEFAULT_CHECKPOINT = {
    "evo2": "evo2_7b_base",
    "ntv3": "InstaDeepAI/NTv3_100M_pre",
    "dnabert2": "zhihan1996/DNABERT-2-117M",
}
# NTv3: -1 is the post-deconv stage, one vector per input token, which is what the
# docstring above promises. -4 is the fourth deconv block and still 8x downsampled;
# see `ntv3_unet.py list --offline` for the full index mapping.
DEFAULT_LAYER = {"evo2": "blocks.26.mlp.l3", "ntv3": -1, "dnabert2": -1}


def embed(
    out,
    model="evo2",
    checkpoint=None,
    layer=None,
    weights=None,
    quartets=QUARTETS,
    limit=0,
    revision="main",
):
    obs = observations(quartets)
    if limit:
        obs = obs.head(limit)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    needed = {}
    for row in obs.itertuples(index=False):
        needed.setdefault(row.wt_id, {"seq": row.wt_seq, "pos": set()})["pos"].add(row.index)
        needed.setdefault(row.mut_id, {"seq": row.mut_seq, "pos": set()})["pos"].add(row.index)

    checkpoint = checkpoint or DEFAULT_CHECKPOINT[model]
    layer = DEFAULT_LAYER[model] if layer is None else layer
    kwargs = {"checkpoint": checkpoint, "layer": layer}
    if model == "evo2" and weights:
        kwargs["weights"] = weights
    if model in ("ntv3", "dnabert2"):
        kwargs["revision"] = revision
    adapter = ADAPTERS[model](**kwargs)
    print(
        f"{adapter.label}: {len(obs)} observations, {len(needed)} distinct sequences",
        flush=True,
    )

    pooled, at_variant, unit_counts = {}, {}, []
    for n, (sequence_id, item) in enumerate(needed.items()):
        h, base_to_unit = adapter.encode(item["seq"])
        # Plain mean over the model's own units. For DNABERT-2 that is a mean
        # over ~40 tokens, not over 200 bases, so no token is weighted by how
        # many bases it happens to cover.
        pooled[sequence_id] = h.mean(0).cpu().numpy()
        unit_counts.append(int(h.shape[0]))
        for position in item["pos"]:
            at_variant[(sequence_id, position)] = h[base_to_unit[position]].cpu().numpy()
        if n % 250 == 0:
            print(f"  {n}/{len(needed)}", flush=True)
    print(f"pooled over {adapter.units}s: {np.mean(unit_counts):.1f} per sequence on average")

    rows = list(obs.itertuples(index=False))
    matrices = {
        "d_mean": np.stack([pooled[r.mut_id] - pooled[r.wt_id] for r in rows]),
        "d_pos": np.stack(
            [at_variant[(r.mut_id, r.index)] - at_variant[(r.wt_id, r.index)] for r in rows]
        ),
        "wt_mean": np.stack([pooled[r.wt_id] for r in rows]),
    }
    for name, matrix in matrices.items():
        np.save(out / f"X_{name}.npy", matrix)
    obs.drop(columns=["wt_seq", "mut_seq"]).to_csv(out / "observations.csv", index=False)
    obs[["wt_id", "wt_seq", "mut_id", "mut_seq", "position", "index"]].to_csv(
        out / "sequences.csv", index=False
    )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model": model,
                "checkpoint": checkpoint,
                "layer": str(layer),
                "revision": revision,
                "source_sha256": source_hashes(),
                "weights_sha256": file_hash(weights) if weights else None,
                "pooling_protocol": POOLING_PROTOCOL
                if model != "dnabert2"
                else "native-real-token-v2",
                "label": adapter.label,
                "units": adapter.units,
                "units_per_sequence": float(np.mean(unit_counts)),
                "n_observations": len(obs),
                "n_sequences": len(needed),
                "width": int(matrices["d_mean"].shape[1]),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote three matrices to {out}, width {matrices['d_mean'].shape[1]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="GPU. One forward pass per distinct sequence.")
    e.add_argument("--model", choices=tuple(ADAPTERS), default="evo2")
    e.add_argument("--out", required=True)
    e.add_argument("--checkpoint", default=None)
    e.add_argument(
        "--layer",
        default=None,
        help="module name for evo2, integer index for ntv3 and dnabert2",
    )
    e.add_argument("--weights", default=None, help="evo2 only")
    e.add_argument("--revision", default="main", help="ntv3/dnabert2; pin the checkpoint commit")
    e.add_argument("--quartets", default=QUARTETS)
    e.add_argument("--limit", type=int, default=0, help="first N observations, smoke run")

    p = sub.add_parser("probe", help="CPU. One model.")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--target", choices=("magnitude", "signed"), default="magnitude")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="fold assignment; vary it to see how much the split matters",
    )

    c = sub.add_parser("compare", help="CPU. Several models side by side.")
    c.add_argument("--embeddings", nargs="+", required=True)
    c.add_argument("--target", choices=("magnitude", "signed"), default="magnitude")
    c.add_argument("--folds", type=int, default=5)
    c.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="fold assignment; vary it to see how much the split matters",
    )

    args = vars(parser.parse_args())
    cmd = args.pop("cmd")
    if cmd == "embed" and args["layer"] is not None and args["model"] != "evo2":
        args["layer"] = int(args["layer"])
    {"embed": embed, "probe": probe, "compare": compare}[cmd](**args)


if __name__ == "__main__":
    main()
