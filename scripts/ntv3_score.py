#!/usr/bin/env python3
"""Score the quartet sequences with Nucleotide Transformer v3.

Writes the same CSV that ``evo_epistasis.py evaluate`` consumes, so the NTv3
result runs through the identical recoding, noise ceiling, detection AUROC,
precision stratification and element baselines as the Evo 2 result.

    python3 ntv3_score.py --quartets data/quartets.csv.gz \
        --checkpoint NTv3_100M_pre --revision main \
        --output results/ntv3_100m_pre/ntv3_scores.csv

    python3 evo_epistasis.py evaluate --quartets data/quartets.csv.gz \
        --scores results/ntv3_100m_pre/ntv3_scores.csv \
        --out results/ntv3_100m_pre --label "NTv3 100M pre"

Requires a Hugging Face login, because the InstaDeepAI checkpoints are gated,
and ``trust_remote_code=True``, because NTv3 ships its own model class.
"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import atomic_json, source_hashes
from benchmark_data import file_hash, load_quartets, reverse_complement
from evo_epistasis import CONTRAST, STATES

SCORE_DEFINITION = (
    "mean(forward,RC) of summed masked-LM pseudo-log-likelihood over the real bases; "
    "one mask per position; N-padded to a multiple of 128; FP32 log-softmax; FP64 sum"
)


class NTv3Scorer:
    """Masked-LM pseudo-log-likelihood from Nucleotide Transformer v3.

    NTv3 is a masked language model, not autoregressive, so there is no
    next-token score to sum.  The comparable quantity is the pseudo-log-
    likelihood: mask one position at a time and sum log P(observed base |
    every other base).  Every position therefore sees both variants of a pair,
    which is the property Evo 2's causal scoring cannot have.

    Two NTv3 constraints are handled here.  Its U-Net needs a sequence length
    divisible by 2**num_downsamples (128 for the 7-downsample models), so a
    200 bp oligo is padded symmetrically to 256.  The padding uses N, not the
    pad token, because the models were not trained on pad tokens.  Only the
    200 real positions are scored.
    """

    MULTIPLE = 128
    POSITION_CHUNK = 100

    from model_runtime import pick_device as pick_device

    pick_device = staticmethod(pick_device)

    def __init__(
        self,
        model_name="NTv3_100M_pre",
        revision=None,
        device="auto",
        use_bfloat16=False,
    ):
        import torch
        from model_runtime import load_hf

        if use_bfloat16:
            raise ValueError("NTv3 requires float32 weights; use --fp32")
        self.torch, self.model_name = torch, model_name
        self.tokenizer, self.model, self.device = load_hf(model_name, revision, device)
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            raise ValueError(
                "NTv3 tokenizer exposes no mask token; cannot compute a pseudo-likelihood"
            )

    from model_runtime import pad_to_multiple as _pad

    _pad = staticmethod(_pad)

    def _pseudo_log_likelihood(self, sequence):
        torch = self.torch
        from model_runtime import tokenize_ntv3

        batch_ids, left, _ = tokenize_ntv3(self.tokenizer, [sequence], self.device)
        ids = batch_ids[0]
        positions = list(range(left, left + len(sequence)))
        truth = ids[positions].clone()
        recovered = "".join(self.tokenizer.convert_ids_to_tokens(truth.tolist()))
        if recovered != sequence:
            raise ValueError(
                "NTv3 token alignment is wrong: the positions being masked do not "
                f"decode back to the input sequence ({recovered[:20]} vs {sequence[:20]})"
            )
        total = 0.0
        for start in range(0, len(positions), self.POSITION_CHUNK):
            chunk = positions[start : start + self.POSITION_CHUNK]
            batch = ids.unsqueeze(0).repeat(len(chunk), 1)
            batch[
                torch.arange(len(chunk), device=self.device),
                torch.tensor(chunk, device=self.device),
            ] = self.mask_id
            with torch.inference_mode():
                output = self.model(input_ids=batch)
                logits = output.logits if hasattr(output, "logits") else output[0]
            if logits.shape[:2] != batch.shape:
                raise ValueError(f"Unexpected NTv3 logits shape {tuple(logits.shape)}")
            logp = torch.log_softmax(logits.float(), -1)
            rows = torch.arange(len(chunk), device=self.device)
            picked = logp[
                rows,
                torch.tensor(chunk, device=self.device),
                truth[start : start + len(chunk)],
            ]
            total += float(picked.detach().cpu().double().sum())  # MPS has no float64
        return total

    def forward(self, sequences):
        if len({len(s) for s in sequences}) != 1:
            raise ValueError("Scoring requires equal lengths")
        scores = np.array([self._pseudo_log_likelihood(s) for s in sequences], float)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite NTv3 score")
        return scores


def score_quartets_ntv3(
    quartets_path,
    output_path,
    checkpoint="NTv3_100M_pre",
    revision="UNRECORDED",
    batch_size=1,
    device="auto",
    use_bfloat16=False,
    limit=0,
):
    """Score each unique sequence once; the output CSV is also the resumable cache."""
    if revision == "UNRECORDED":
        raise ValueError("Record the exact checkpoint snapshot with --revision")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    quartets = load_quartets(quartets_path)
    sequences = (
        pd.concat(
            [
                quartets[[f"id_{s}", f"seq_{s}"]].set_axis(["sequence_id", "sequence"], axis=1)
                for s in STATES
            ]
        )
        .drop_duplicates()
        .sort_values("sequence_id")
    )
    if sequences.sequence_id.duplicated().any():
        raise ValueError("One sequence hash maps to multiple sequences")
    if limit:
        sequences = sequences.head(
            limit
        )  # smoke test; meta records it so a full run needs a new path
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.stem + ".meta.json")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "torch", "transformers")
    }
    meta = {
        "backend": "ntv3",
        "checkpoint": checkpoint,
        "revision": revision,
        "contrast": CONTRAST,
        "score": SCORE_DEFINITION,
        "quartets_sha256": file_hash(quartets_path),
        "code_sha256": file_hash(__file__),
        "source_sha256": source_hashes(),
        "batch_size": batch_size,
        "bfloat16": use_bfloat16,
        "limit": limit or None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Scoring configuration changed; use a new output path")
    if output.exists() and not meta_path.exists():
        raise ValueError("score cache has no provenance; use a new output path")
    atomic_json(meta_path, meta)

    columns = ["sequence_id", "forward", "reverse", "score"]
    cached = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=columns)
    if cached.sequence_id.duplicated().any() or (
        len(cached) and not np.isfinite(cached[columns[1:]]).all().all()
    ):
        raise ValueError("Invalid score cache")
    pending = sequences[~sequences.sequence_id.isin(cached.sequence_id)]
    scorer = NTv3Scorer(checkpoint, revision, device, use_bfloat16) if len(pending) else None
    if scorer:
        print(
            f"scoring {len(pending)} of {len(sequences)} sequences on {scorer.device}",
            flush=True,
        )
    for _, length_group in pending.groupby(pending.sequence.str.len(), sort=True):
        for start in range(0, len(length_group), batch_size):
            batch = length_group.iloc[start : start + batch_size]
            seqs = batch.sequence.tolist()
            forward = scorer.forward(seqs)
            reverse = scorer.forward([reverse_complement(s) for s in seqs])
            rows = pd.DataFrame(
                {
                    "sequence_id": batch.sequence_id,
                    "forward": forward,
                    "reverse": reverse,
                    "score": (forward + reverse) / 2,
                }
            )
            rows.to_csv(output, mode="a", header=not output.exists(), index=False)
    scores = (
        pd.read_csv(output)
        .drop_duplicates("sequence_id")
        .set_index("sequence_id")
        .loc[sequences.sequence_id]
        .reset_index()
    )
    scores.to_csv(output, index=False)
    return {"sequences": len(scores), "newly_scored": len(pending), **meta}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quartets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default="NTv3_100M_pre",
        help="NTv3 model name, or a full org/name. Use a _pre checkpoint: the _post "
        "models were supervised on ~16,000 functional tracks, which may include K562.",
    )
    parser.add_argument(
        "--revision", default="UNRECORDED", help="Hugging Face revision or commit sha"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="sequences per step; positions are batched inside",
    )
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N unique sequences, for a smoke test. "
        "Use a different --output than the full run.",
    )
    parser.add_argument(
        "--fp32", action="store_true", help="compatibility flag; float32 is always used"
    )
    args = parser.parse_args()
    result = score_quartets_ntv3(
        args.quartets,
        args.output,
        args.checkpoint,
        args.revision,
        args.batch_size,
        args.device,
        False,
        args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
