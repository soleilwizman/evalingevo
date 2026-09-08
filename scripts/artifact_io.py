"""Validation and durable provenance for benchmark artifacts."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from benchmark_data import file_hash


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_hashes():
    root = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}


def aligned_tables(left, right, columns, *, numeric_columns=()):
    for table in (left, right):
        if not set(columns) <= set(table) or table[columns].isna().any().any():
            raise ValueError(f"missing or null comparison columns: {columns}")
    if len(left) != len(right):
        raise ValueError("comparison tables have different row counts")
    for column in columns:
        a, b = left[column].to_numpy(), right[column].to_numpy()
        # Only measured values may differ by CSV round-trip noise. IDs and
        # group/index columns remain exact, even when they are numeric.
        equal = (
            np.allclose(a, b, rtol=0, atol=1e-12)
            if column in numeric_columns
            else np.array_equal(a, b)
        )
        if not equal:
            raise ValueError(
                f"comparison table differs in {column}; rows must be identically ordered"
            )


def load_matrix(path, table, keys):
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError("matrix row identifiers must be complete and unique")
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(table) or not matrix.shape[1]:
        raise ValueError(f"{path}: embedding dimensions do not match the row table")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: non-finite embeddings; re-embed instead of dropping rows")
    return matrix


def matched_embedding_metadata(variant_dir, reference_dir):
    variant, reference = [
        json.loads((Path(d) / "meta.json").read_text()) for d in (variant_dir, reference_dir)
    ]
    for field in ("checkpoint", "pooling_protocol"):
        if not variant.get(field) or variant.get(field) != reference.get(field):
            raise ValueError(f"reference and alternate embeddings must record matching {field}")
    if variant.get("weights_sha256") or reference.get("weights_sha256"):
        if variant.get("weights_sha256") != reference.get("weights_sha256"):
            raise ValueError("reference and alternate local weight hashes must match")
    elif not variant.get("revision") or variant.get("revision") != reference.get("revision"):
        raise ValueError("reference and alternate embeddings must record matching revision")

    def representation(metadata):
        return str(metadata.get("representation", metadata.get("layer")))

    if representation(variant) != representation(reference):
        raise ValueError("reference and alternate embeddings describe different representations")
    return variant, reference
