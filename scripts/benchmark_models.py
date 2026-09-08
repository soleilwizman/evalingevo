"""Prespecified models and element representations for the primary comparison."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    label: str
    family: str
    checkpoint: str
    layer: str | int
    representation: str
    score_directory: str

    def check_revision(self, revision):
        if self.family != "evo2" and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError(
                "primary HF embeddings require a full immutable 40-character commit SHA"
            )


MODELS = {
    "evo2": ModelSpec(
        "Evo 2 7B", "evo2", "evo2_7b_base", "blocks.26.mlp.l3", "blocks.26.mlp.l3", "evo2_7b_base"
    ),
    "ntv3_100m": ModelSpec(
        "NTv3 100M", "ntv3", "InstaDeepAI/NTv3_100M_pre", -1, "final_deconvolution", "ntv3_100m_pre"
    ),
    "ntv3_650m": ModelSpec(
        "NTv3 650M", "ntv3", "InstaDeepAI/NTv3_650M_pre", -1, "final_deconvolution", "ntv3_650m_pre"
    ),
    "dnabert2": ModelSpec(
        "DNABERT-2",
        "dnabert2",
        "zhihan1996/DNABERT-2-117M",
        -1,
        "last_encoder_layer",
        "dnabert2_117m",
    ),
}
