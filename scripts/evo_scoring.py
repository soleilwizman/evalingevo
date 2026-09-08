"""Evo autoregressive scoring and resumable score-cache orchestration."""

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_io import atomic_json, source_hashes
from benchmark_data import (
    CONTRAST,
    STATES,
    file_hash,
    load_quartets,
    reverse_complement,
)


class EvoScorer:
    def __init__(self, checkpoint, weights=None, device="cuda:0"):
        import torch
        from model_runtime import load_evo

        self.torch = torch
        self.model, self.device = load_evo(checkpoint, weights, device)

    def forward(self, sequences):
        torch, tokenizer = self.torch, self.model.tokenizer
        tokens = [tokenizer.tokenize(s) for s in sequences]
        if len({len(s) for s in sequences}) != 1 or any(
            len(t) != len(s) for t, s in zip(tokens, sequences)
        ):
            raise ValueError("Scoring requires equal lengths and one token per nucleotide")
        ids = torch.tensor(
            [[tokenizer.eod_id] + t for t in tokens],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            output = self.model(ids)[0]
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if logits.ndim != 3 or logits.shape[:2] != ids.shape:
                raise ValueError("Unexpected Evo output shape")
            logp = torch.log_softmax(logits[:, :-1].float(), -1)
            scores = logp.gather(-1, ids[:, 1:, None]).squeeze(-1).double().sum(-1).cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite Evo score")
        return scores


def score_quartets(
    quartets_path,
    output_path,
    backend="evo",
    checkpoint="evo2_7b_base",
    revision="UNRECORDED",
    batch_size=1,
    weights=None,
):
    """Score each unique sequence once; the output CSV is also the resumable cache."""
    if backend not in {"evo", "gc"} or batch_size < 1:
        raise ValueError("backend must be evo or gc, and batch_size must be positive")
    if backend == "evo" and revision == "UNRECORDED":
        raise ValueError("Record the exact checkpoint snapshot with --revision")
    quartets = load_quartets(quartets_path)
    sequences = pd.concat(
        [
            quartets[[f"id_{s}", f"seq_{s}"]].set_axis(["sequence_id", "sequence"], axis=1)
            for s in STATES
        ]
    )
    sequences = sequences.drop_duplicates().sort_values("sequence_id")
    if sequences.sequence_id.duplicated().any():
        raise ValueError("One sequence hash maps to multiple sequences")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.stem + ".meta.json")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
    }
    score_definition = (
        "mean(forward,RC) of summed AR log-likelihood; EOD-as-BOS; FP32 log-softmax; FP64 sum"
        if backend == "evo"
        else "GC count; additive negative control"
    )
    meta = {
        "backend": backend,
        "checkpoint": checkpoint if backend == "evo" else None,
        "revision": revision if backend == "evo" else None,
        "contrast": CONTRAST,
        "score": score_definition,
        "quartets_sha256": file_hash(quartets_path),
        "code_sha256": file_hash(__file__),
        "source_sha256": source_hashes(),
        "weights_sha256": file_hash(weights) if weights else None,
        "batch_size": batch_size,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Scoring configuration changed; use a new output path")
    if output.exists() and not meta_path.exists():
        raise ValueError("score cache has no provenance; use a new output path")
    atomic_json(meta_path, meta)
    cached = (
        pd.read_csv(output)
        if output.exists()
        else pd.DataFrame(columns=["sequence_id", "forward", "reverse", "score"])
    )
    if cached.sequence_id.duplicated().any() or (
        len(cached) and not np.isfinite(cached[["forward", "reverse", "score"]]).all().all()
    ):
        raise ValueError("Invalid score cache")
    pending = sequences[~sequences.sequence_id.isin(cached.sequence_id)]
    scorer = EvoScorer(checkpoint, weights) if backend == "evo" and len(pending) else None
    for _, length_group in pending.groupby(pending.sequence.str.len(), sort=True):
        for start in range(0, len(length_group), batch_size):
            batch = length_group.iloc[start : start + batch_size]
            seqs = batch.sequence.tolist()
            if scorer:
                forward, reverse = (
                    scorer.forward(seqs),
                    scorer.forward([reverse_complement(s) for s in seqs]),
                )
            else:
                forward = reverse = np.array([s.count("G") + s.count("C") for s in seqs], float)
            rows = pd.DataFrame(
                {
                    "sequence_id": batch.sequence_id,
                    "forward": forward,
                    "reverse": reverse,
                    "score": (np.asarray(forward) + np.asarray(reverse)) / 2,
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
