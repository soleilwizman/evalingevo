"""Single-variant measurements, measurement uncertainty and within-element ranks."""

from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_data import load_quartets

QUARTETS = "data/quartets.csv.gz"
AUDIT = "data/audit.csv.gz"


def observations(quartets_path=QUARTETS):
    """One row per single-variant measurement: two per quartet, stacked."""
    q = load_quartets(quartets_path)
    frames = []
    for side in ("a", "b"):
        frames.append(
            pd.DataFrame(
                {
                    "wt_id": q.id_wt,
                    "mut_id": q[f"id_{side}"],
                    "wt_seq": q.seq_wt,
                    "mut_seq": q[f"seq_{side}"],
                    "position": q[f"pos_{side}"],
                    "effect": q[f"y_{side}"],
                    "group": q.group_id,
                    "side": side,
                }
            )
        )
    obs = pd.concat(frames, ignore_index=True)
    # pos_a and pos_b are 1-based, so the changed base sits at position - 1.
    # The check below refuses to run if that is ever untrue, because reading one
    # base off the variant produces a clean and entirely plausible null.
    obs["index"] = obs.position - 1
    bad = [
        r.wt_seq[r.index] == r.mut_seq[r.index]
        or sum(a != b for a, b in zip(r.wt_seq, r.mut_seq)) != 1
        for r in obs.itertuples(index=False)
    ]
    if any(bad):
        raise SystemExit(
            f"{sum(bad)} of {len(obs)} rows do not differ by exactly one "
            "base at the stated position; the position convention is wrong"
        )
    # 207 variants are measured in more than one quartet and those repeats
    # disagree, by up to 0.20 in effect. Keeping whichever came first would be an
    # arbitrary choice between two real measurements, so they are averaged.
    keys = ["wt_id", "mut_id", "index"]
    if (obs.groupby("wt_id").group.nunique() > 1).any():
        raise ValueError("reference sequences span multiple genomic groups")
    repeats = obs.groupby(keys).effect.transform("size")
    averaged = obs.groupby(keys, as_index=False).agg(
        wt_seq=("wt_seq", "first"),
        mut_seq=("mut_seq", "first"),
        position=("position", "first"),
        effect=("effect", "mean"),
        group=("group", "first"),
        side=("side", "first"),
        replicates=("effect", "size"),
    )
    if repeats.max() > 1:
        print(
            f"{int((repeats > 1).sum())} rows are repeated measurements of "
            f"{int((averaged.replicates > 1).sum())} variants; averaged",
            flush=True,
        )
    return averaged.reset_index(drop=True)


def noise_ceiling(obs, quartets_path=QUARTETS, audit_path=AUDIT):
    """How high can any predictor of the SIGNED effect reach here?

    Each single-variant measurement carries a standard error, so part of the
    spread in the target is assay noise that nothing can predict. Reliability is
    (Var(y) - mean SE^2) / Var(y), and its square root bounds the correlation.

    Returns None if the audit table is missing, since it is optional.

    Only defined for the signed target: attenuation is a linear-model result and
    taking absolute values is not a linear operation, so the magnitude target has
    no ceiling from this formula.
    """
    if not Path(audit_path).exists():
        return None
    q = load_quartets(quartets_path)
    audit = pd.read_csv(audit_path)
    key = ["v1", "v2", "center_variant", "window", "library"]
    if any(c not in audit.columns for c in key):
        return None
    joined = audit[key[0]].astype(str)
    for column in key[1:]:
        joined = joined + ";" + audit[column].astype(str)
    audit = audit.assign(_k=joined)
    wanted = ["_k", "altref_log2SkewSE", "refalt_log2SkewSE"]
    if any(c not in audit.columns for c in wanted):
        return None
    q = q.assign(_k=q.pair_id.str.split("|").str[0]).merge(
        audit.drop_duplicates("_k")[wanted], on="_k", how="left"
    )
    se = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "wt_id": q.id_wt,
                        "mut_id": q.id_a,
                        "index": q.pos_a - 1,
                        "se": q.altref_log2SkewSE,
                    }
                ),
                pd.DataFrame(
                    {
                        "wt_id": q.id_wt,
                        "mut_id": q.id_b,
                        "index": q.pos_b - 1,
                        "se": q.refalt_log2SkewSE,
                    }
                ),
            ]
        )
        .groupby(["wt_id", "mut_id", "index"], as_index=False)
        .se.mean()
    )
    merged = obs.merge(se, on=["wt_id", "mut_id", "index"], how="left")
    if merged.se.isna().any():
        return None
    variance = float(np.var(merged.effect, ddof=0))
    noise = float(np.mean(merged.se**2))
    reliability = (variance - noise) / variance
    return {
        "variance": variance,
        "mean_se_squared": noise,
        "reliability": reliability,
        "ceiling": float(np.sqrt(max(reliability, 0.0))),
    }


def within_element_accuracy(prediction, y, groups):
    """Can it rank the variants INSIDE an element?

    The overall Spearman mixes two things: telling a fragile element from a
    robust one, and telling which base inside an element matters. Only the
    second is variant interpretation, and it is the smaller of the two here,
    since roughly half the variance in effect size sits between elements.

    Every within-element pair is scored on whether the predicted ordering
    matches the measured one. Chance is 0.500 by construction: no amount of
    knowing the element can help, because both members share it.
    """
    frame = pd.DataFrame({"p": prediction, "y": y, "g": groups})
    correct = total = 0
    for _, block in frame.groupby("g"):
        if len(block) < 2:
            continue
        p_values, y_values = block.p.to_numpy(), block.y.to_numpy()
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if y_values[i] == y_values[j]:
                    continue
                total += 1
                if p_values[i] == p_values[j]:
                    correct += 0.5
                else:
                    correct += (p_values[i] > p_values[j]) == (y_values[i] > y_values[j])
    return (correct / total if total else float("nan")), total
