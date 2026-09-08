"""Canonical element table shared by all reference-activity probes."""

import pandas as pd
from artifact_io import aligned_tables

PRED = "results/evo2_7b_base/predictions.csv"
AUDIT = "data/audit.csv.gz"


def elements(pred=PRED, audit=AUDIT):
    p = pd.read_csv(pred)
    a = pd.read_csv(audit)
    a["key"] = (
        a.v1.astype(str)
        + ";"
        + a.v2.astype(str)
        + ";"
        + a.center_variant.astype(str)
        + ";"
        + a.window.astype(str)
        + ";"
        + a.library.astype(str)
    )
    p["key"] = p.pair_id.str.split("|").str[0]
    m = p.merge(a[["key", "refref_Log2FC", "refref_active"]], on="key", how="inner")
    if len(m) != len(p):
        raise ValueError(f"audit join dropped rows: {len(m)} of {len(p)}")

    # A reference sequence must not straddle two cross-validation groups, or
    # near-identical sequence lands on both sides of a fold boundary.
    spread = m.groupby("id_wt").group_id.nunique()
    if (spread > 1).any():
        raise ValueError(
            f"{(spread > 1).sum()} reference sequences span "
            "multiple region groups; grouping is unsafe"
        )

    el = (
        m.groupby("id_wt")
        .agg(
            seq=("seq_wt", "first"),
            s_wt=("s_wt", "first"),
            activity=("refref_Log2FC", "mean"),
            active=("refref_active", "max"),
            group=("group_id", "first"),
        )
        .dropna(subset=["activity"])
        .reset_index()
        .rename(columns={"id_wt": "sequence_id"})
    )
    if el.seq.str.len().nunique() != 1:
        raise ValueError("reference sequences are not all the same length")
    if el.seq.duplicated().any():
        raise ValueError("two sequence ids share a sequence")
    return el


def validate_elements(table, full):
    if table.sequence_id.duplicated().any():
        raise ValueError("duplicate embedded element IDs")
    if not table.sequence_id.isin(full.index).all():
        raise ValueError("embedded sequences are missing from the current element table")
    current = full.loc[table.sequence_id].reset_index()
    aligned_tables(
        table, current, ["sequence_id", "activity", "group"], numeric_columns=("activity",)
    )
