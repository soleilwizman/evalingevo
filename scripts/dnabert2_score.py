#!/usr/bin/env python3
"""Score the quartet sequences with DNABERT-2 (zhihan1996/DNABERT-2-117M).

Writes the same CSV that ``evo_epistasis.py evaluate`` consumes, so the result
runs through the identical recoding, noise ceiling, detection AUROC and element
baselines as the Evo 2 and NTv3 results.

    python3 scripts/dnabert2_score.py --quartets data/quartets.csv.gz \
        --checkpoint zhihan1996/DNABERT-2-117M --revision <commit sha> \
        --output results/dnabert2_117m/dnabert2_scores.csv

    python3 scripts/evo_epistasis.py evaluate --quartets data/quartets.csv.gz \
        --scores results/dnabert2_117m/dnabert2_scores.csv \
        --out results/dnabert2_117m --label "DNABERT-2 117M"

    python3 scripts/single_variant.py --predictions results/dnabert2_117m/predictions.csv \
        --out results/dnabert2_117m/single_variant.json --label "DNABERT-2 117M"

DNABERT-2 is a masked language model over a byte-pair vocabulary, so the score is
the pseudo-log-likelihood summed over its tokens: mask one token at a time and add
log P(observed token | every other token). Evo 2 and NTv3 sum over the 200 bases;
DNABERT-2 sums over roughly 40 tokens of four or five bases each, and a single
substitution can move the token boundaries around it. The score is therefore in
the model's own units and not the same quantity as the per-base ones. It is still
one number per sequence, which is all that the quartet contrast needs.

The checkpoint ships its own model class (MosaicBERT with ALiBi and unpadding),
targets transformers 4.x and needs ``einops``. Load with
``trust_remote_code=True``. Its Triton attention kernel is optional and the code
falls back to plain PyTorch attention on CPU or when Triton is absent.

The masked-LM head is checked after loading: if any ``cls.*`` weight is missing
from the checkpoint the head is randomly initialised, and a pseudo-likelihood
from it would be noise, so the run refuses.
"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from evo_epistasis import CONTRAST, STATES, file_hash, load_quartets, reverse_complement

SCORE_DEFINITION = ("mean(forward,RC) of summed masked-LM pseudo-log-likelihood over the BPE tokens "
                    "of the unpadded sequence; one mask per token; FP32 log-softmax; FP64 sum")


class DNABERT2Scorer:
    """Masked-LM pseudo-log-likelihood from DNABERT-2, one token masked per copy.

    One sequence at a time: its masked copies all have the same length, so no
    padding or attention mask is needed and the unpadding code path in the
    checkpoint's model class is never exercised on padded input.
    """

    TOKEN_CHUNK = 64

    @staticmethod
    def pick_device(requested="auto"):
        import torch
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __init__(self, checkpoint="zhihan1996/DNABERT-2-117M", revision=None, device="auto"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch, self.checkpoint = torch, checkpoint
        self.device = self.pick_device(device)
        kwargs = {"trust_remote_code": True}
        if revision and revision != "UNRECORDED":
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
        self.model, info = AutoModelForMaskedLM.from_pretrained(
            checkpoint, output_loading_info=True, **kwargs)
        head_missing = sorted(k for k in info.get("missing_keys", []) if k.startswith("cls."))
        if head_missing:
            raise ValueError("the checkpoint carries no masked-LM head, so the pseudo-likelihood "
                             f"would come from random weights; missing {head_missing[:4]}")
        self.model = self.model.float().eval().to(self.device)
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            raise ValueError("tokenizer exposes no mask token; cannot compute a pseudo-likelihood")
        self.specials = set(self.tokenizer.all_special_ids)

    def _tokens(self, sequence):
        """Token ids for one sequence, and the positions of its real tokens.

        The real tokens must spell the input back exactly; a vocabulary that
        drops or rewrites bases would silently score a different sequence.
        """
        ids = self.tokenizer(sequence, add_special_tokens=True)["input_ids"]
        real = [i for i, t in enumerate(ids) if t not in self.specials]
        pieces = self.tokenizer.convert_ids_to_tokens([ids[i] for i in real])
        recovered = "".join(p.replace("##", "") for p in pieces)
        if recovered != sequence:
            raise ValueError("DNABERT-2 tokens do not spell the input back "
                             f"({recovered[:24]}... vs {sequence[:24]}...); "
                             "the tokenizer is not a plain BPE over ACGT")
        return ids, real

    def _pseudo_log_likelihood(self, sequence):
        torch = self.torch
        ids, real = self._tokens(sequence)
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        truth = ids[real].clone()
        total = 0.0
        for start in range(0, len(real), self.TOKEN_CHUNK):
            chunk = real[start:start + self.TOKEN_CHUNK]
            batch = ids.unsqueeze(0).repeat(len(chunk), 1)
            rows = torch.arange(len(chunk), device=self.device)
            cols = torch.tensor(chunk, device=self.device)
            batch[rows, cols] = self.mask_id
            with torch.inference_mode():
                output = self.model(input_ids=batch)
                logits = output.logits if hasattr(output, "logits") else output[0]
            if logits.shape[:2] != batch.shape:
                raise ValueError(f"Unexpected DNABERT-2 logits shape {tuple(logits.shape)} "
                                 f"for input {tuple(batch.shape)}")
            logp = torch.log_softmax(logits.float(), -1)
            picked = logp[rows, cols, truth[start:start + len(chunk)]]
            total += float(picked.detach().cpu().double().sum())
        return total

    def forward(self, sequences):
        scores = np.array([self._pseudo_log_likelihood(s) for s in sequences], float)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite DNABERT-2 score")
        return scores


def score_quartets_dnabert2(quartets_path, output_path, checkpoint="zhihan1996/DNABERT-2-117M",
                            revision="UNRECORDED", device="auto", limit=0):
    """Score each unique sequence once; the output CSV is also the resumable cache."""
    if revision == "UNRECORDED":
        raise ValueError("Record the exact checkpoint snapshot with --revision")
    quartets = load_quartets(quartets_path)
    sequences = pd.concat([quartets[[f"id_{s}", f"seq_{s}"]].set_axis(["sequence_id", "sequence"], axis=1)
                           for s in STATES]).drop_duplicates().sort_values("sequence_id")
    if sequences.sequence_id.duplicated().any():
        raise ValueError("One sequence hash maps to multiple sequences")
    if limit:
        sequences = sequences.head(limit)  # smoke test; meta records it so a full run needs a new path
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.stem + ".meta.json")
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "torch", "transformers", "einops")}
    meta = {"backend": "dnabert2", "checkpoint": checkpoint, "revision": revision, "contrast": CONTRAST,
            "score": SCORE_DEFINITION, "quartets_sha256": file_hash(quartets_path),
            "code_sha256": file_hash(__file__), "limit": limit or None,
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions}
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Scoring configuration changed; use a new output path")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    columns = ["sequence_id", "forward", "reverse", "score"]
    cached = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=columns)
    if cached.sequence_id.duplicated().any() or (len(cached) and not np.isfinite(cached[columns[1:]]).all().all()):
        raise ValueError("Invalid score cache")
    pending = sequences[~sequences.sequence_id.isin(cached.sequence_id)]
    scorer = DNABERT2Scorer(checkpoint, revision, device) if len(pending) else None
    if scorer:
        print(f"scoring {len(pending)} of {len(sequences)} sequences on {scorer.device}", flush=True)
    for start in range(0, len(pending), 16):
        batch = pending.iloc[start:start + 16]
        seqs = batch.sequence.tolist()
        forward = scorer.forward(seqs)
        reverse = scorer.forward([reverse_complement(s) for s in seqs])
        rows = pd.DataFrame({"sequence_id": batch.sequence_id, "forward": forward, "reverse": reverse,
                             "score": (forward + reverse) / 2})
        rows.to_csv(output, mode="a", header=not output.exists(), index=False)
    scores = (pd.read_csv(output).drop_duplicates("sequence_id").set_index("sequence_id")
                .loc[sequences.sequence_id].reset_index())
    scores.to_csv(output, index=False)
    return {"sequences": len(scores), "newly_scored": len(pending), **meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quartets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="zhihan1996/DNABERT-2-117M",
                        help="Hugging Face repo id, or a local directory holding the checkpoint")
    parser.add_argument("--revision", default="UNRECORDED", help="Hugging Face revision or commit sha")
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N unique sequences, for a smoke test. "
                             "Use a different --output than the full run.")
    args = parser.parse_args()
    result = score_quartets_dnabert2(args.quartets, args.output, args.checkpoint, args.revision,
                                     args.device, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
