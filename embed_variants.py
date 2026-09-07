#!/usr/bin/env python3
"""GPU: embed the variant sequences, not just the reference 200-mers.

The committed embeddings cover the 2,595 distinct reference sequences and none
of the 5,428 variant sequences, so no probe of a variant effect is possible.
This embeds reference and variant sequences together in one pass, which is the
only way the difference vector h(alt) - h(ref) is guaranteed to come from one
checkpoint, one dtype and one code path.

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

from evo_epistasis import STATES, file_hash, load_quartets

MULTIPLE = 128
POOLINGS = ("mean", "last")


def sequence_table(quartets_path, include_double=False):
    """One row per distinct sequence: the references and every single variant."""
    quartets = load_quartets(quartets_path)
    wanted = ("wt", "a", "b") + (("ab",) if include_double else ())
    frames = []
    for state in wanted:
        frames.append(pd.DataFrame({
            "sequence_id": quartets[f"id_{state}"],
            "seq": quartets[f"seq_{state}"],
            "role": state,
            "group_id": quartets.group_id,
            "pair_id": quartets.pair_id}))
    table = pd.concat(frames, ignore_index=True)
    # A sequence can appear in several pairs; embed it once.
    table = (table.sort_values(["sequence_id", "pair_id"])
                  .drop_duplicates("sequence_id")
                  .sort_values("sequence_id")
                  .reset_index(drop=True))
    if table.seq.str.len().nunique() != 1:
        raise SystemExit("sequences are not all the same length")
    if table.sequence_id.duplicated().any():
        raise SystemExit("duplicate sequence ids after dedup")
    return table


class Evo2Embedder:
    """Mirrors evo_probe.embed so variant rows are comparable with the committed ones."""

    def __init__(self, checkpoint, layer, weights=None):
        import torch
        from evo2 import Evo2
        if not torch.cuda.is_available():
            raise SystemExit("Evo 2 embedding requires an NVIDIA GPU; torch.cuda "
                             "reports no device")
        self.torch, self.layer = torch, layer
        self.model = Evo2(checkpoint, local_path=weights)
        self.model.model.eval()
        self.tokenizer = self.model.tokenizer
        try:
            self.model.model.get_submodule(layer)
        except AttributeError:
            names = sorted({n for n, _ in self.model.model.named_modules()
                            if "blocks." in n and n.count(".") <= 3})
            raise SystemExit(f"no submodule '{layer}'. Candidates:\n  " + "\n  ".join(names[:40]))

    @property
    def width_probe(self):
        return None

    def __call__(self, sequences):
        torch = self.torch
        tokens = [self.tokenizer.tokenize(s) for s in sequences]
        if any(len(t) != len(s) for t, s in zip(tokens, sequences)):
            raise SystemExit("tokenizer is not one token per nucleotide")
        ids = torch.tensor([[self.tokenizer.eod_id] + t for t in tokens],
                           dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            output = self.model(ids, return_embeddings=True, layer_names=[self.layer])
        embeddings = output[1] if isinstance(output, (tuple, list)) else output
        hidden = embeddings[self.layer] if isinstance(embeddings, dict) else embeddings
        if hidden.ndim != 3 or hidden.shape[:2] != ids.shape:
            raise SystemExit(f"unexpected embedding shape {tuple(hidden.shape)} "
                             f"for input {tuple(ids.shape)}")
        hidden = hidden[:, 1:, :].float()          # drop the EOD position
        return (hidden.mean(1).cpu().numpy().astype(np.float32),
                hidden[:, -1, :].cpu().numpy().astype(np.float32))


class NTv3Embedder:
    """Mirrors ntv3_probe.embed: N-padded to 256, hook on the block's final layer norm.

    Note for interpretation, not a bug: with 7 downsamples a 256-token input is
    two positions at the transformer, so 'mean' averages two vectors and a
    single-base change has to survive 128x downsampling to appear at all.
    """

    def __init__(self, checkpoint, layer, revision="main"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
        # Cast after loading: NTv3 builds float32 activations internally.
        self.model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs).float()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: no GPU found, running on CPU. This will be slow.", flush=True)
        self.model.eval().to(self.device)
        if not hasattr(self.model, "core") or not hasattr(self.model.core, "transformer_blocks"):
            raise SystemExit("checkpoint does not expose core.transformer_blocks")
        blocks = self.model.core.transformer_blocks
        if not 0 <= layer < len(blocks):
            raise SystemExit(f"--layer must be 0..{len(blocks) - 1} for this checkpoint "
                             f"({len(blocks)} blocks)")
        self.representation = f"core.transformer_blocks.{layer}.final_layer_norm"
        self.captured = {}
        blocks[layer].final_layer_norm.register_forward_hook(
            lambda _m, _i, output: self.captured.__setitem__("h", output))
        self.offset = self._token_offset()

    def _token_offset(self):
        probe = "ACGT" * (MULTIPLE // 4)
        ids = self.tokenizer(probe, add_special_tokens=True)["input_ids"]
        wanted = self.tokenizer.convert_tokens_to_ids(list("ACGT"))
        for start in range(len(ids) - len(probe) + 1):
            if list(ids[start:start + 4]) == list(wanted) and len(ids) - start >= len(probe):
                return start
        raise SystemExit("cannot align NTv3 tokens to bases")

    def _pad(self, sequence):
        target = -(-len(sequence) // MULTIPLE) * MULTIPLE
        extra = target - len(sequence)
        left = extra // 2
        return "N" * left + sequence + "N" * (extra - left), left

    def __call__(self, sequences):
        torch = self.torch
        padded, left = zip(*[self._pad(s) for s in sequences])
        ids = torch.tensor([self.tokenizer(p, add_special_tokens=True)["input_ids"]
                            for p in padded], dtype=torch.long, device=self.device)
        span = slice(self.offset + left[0], self.offset + left[0] + len(sequences[0]))
        decoded = "".join(self.tokenizer.convert_ids_to_tokens(ids[0, span].tolist()))
        if decoded != sequences[0]:
            raise SystemExit(f"token alignment wrong: {decoded[:20]} vs {sequences[0][:20]}")
        self.captured.clear()
        with torch.inference_mode():
            self.model(input_ids=ids)
        hidden = self.captured.get("h")
        if hidden is None:
            raise SystemExit("the layer-norm hook did not fire")
        hidden = hidden.float()
        if hidden.shape[0] != len(sequences):
            raise SystemExit(f"layer output is not batch-first: {tuple(hidden.shape)}")
        if not torch.isfinite(hidden).all():
            raise SystemExit("non-finite activations")
        return (hidden.mean(1).cpu().numpy().astype(np.float32),
                hidden[:, -1, :].cpu().numpy().astype(np.float32))


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = sequence_table(args.quartets, args.include_double)
    print(f"{len(table)} distinct sequences to embed "
          f"({(table.role == 'wt').sum()} reference, "
          f"{(table.role != 'wt').sum()} variant)", flush=True)

    meta_path = out / "meta.json"
    meta = {"backend": args.backend, "checkpoint": args.checkpoint, "layer": str(args.layer),
            "revision": args.revision, "n": int(len(table)),
            "include_double": bool(args.include_double),
            "quartets_sha256": file_hash(args.quartets),
            "code_sha256": file_hash(__file__),
            "python": platform.python_version(), "platform": platform.platform()}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        drift = {k for k in meta if k not in ("platform", "python") and existing.get(k) != meta[k]}
        if drift:
            raise SystemExit(f"this directory holds a different run; fields differ: "
                             f"{sorted(drift)}. Use a new --out.")

    progress_path = out / "progress.json"
    done = json.loads(progress_path.read_text())["done"] if progress_path.exists() else 0
    if done >= len(table):
        print("already complete", flush=True)
        return
    if done:
        print(f"resuming at row {done}", flush=True)

    embedder = (Evo2Embedder(args.checkpoint, args.layer, args.weights)
                if args.backend == "evo2"
                else NTv3Embedder(args.checkpoint, int(args.layer), args.revision))

    memmaps, sequences = {}, table.seq.tolist()
    for start in range(done, len(table), args.batch_size):
        batch = sequences[start:start + args.batch_size]
        pooled = dict(zip(POOLINGS, embedder(batch)))
        for name, block in pooled.items():
            path = out / f"X_{name}.npy"
            if name not in memmaps:
                if path.exists():
                    memmaps[name] = np.lib.format.open_memmap(path, mode="r+")
                    if memmaps[name].shape != (len(table), block.shape[1]):
                        raise SystemExit(f"{path} has shape {memmaps[name].shape}, "
                                         f"expected {(len(table), block.shape[1])}")
                else:
                    memmaps[name] = np.lib.format.open_memmap(
                        path, mode="w+", dtype=np.float32,
                        shape=(len(table), block.shape[1]))
            memmaps[name][start:start + len(batch)] = block
        if (start // args.batch_size) % 25 == 0:
            for m in memmaps.values():
                m.flush()
            progress_path.write_text(json.dumps({"done": start + len(batch)}) + "\n")
            print(f"  {start + len(batch)}/{len(table)}", flush=True)

    for m in memmaps.values():
        m.flush()
    progress_path.write_text(json.dumps({"done": len(table)}) + "\n")
    meta["width"] = int(next(iter(memmaps.values())).shape[1])
    if args.backend == "ntv3":
        meta["representation"] = embedder.representation
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    table.drop(columns=["seq"]).to_csv(out / "sequences.csv", index=False)
    print(f"wrote {out}/X_mean.npy and X_last.npy, width {meta['width']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=("evo2", "ntv3"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--quartets", default="data/quartets.csv.gz")
    parser.add_argument("--checkpoint", default=None,
                        help="evo2_7b_base, or an InstaDeepAI/NTv3_* repo id")
    parser.add_argument("--layer", default=None,
                        help="evo2: a submodule name such as blocks.26.mlp.l3. "
                             "ntv3: a transformer block index, 5 for 100M, 11 for 650M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--weights", default=None, help="evo2 local weight file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--include-double", action="store_true",
                        help="also embed the AB sequences, for a later interaction probe")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = "evo2_7b_base" if args.backend == "evo2" else "InstaDeepAI/NTv3_100M_pre"
    if args.layer is None:
        args.layer = "blocks.26.mlp.l3" if args.backend == "evo2" else "5"
    run(args)


if __name__ == "__main__":
    main()
