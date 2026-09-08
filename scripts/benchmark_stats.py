"""Metrics and sequence baselines independent of inference and plotting."""

from itertools import product

import numpy as np
import pandas as pd
from benchmark_data import KMER_VOCAB
from scipy.stats import pearsonr, spearmanr


def correlations(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) < 3 or np.ptp(y) < 1e-12 or np.ptp(pred) < 1e-12:
        return {"spearman": None, "pearson": None}
    return {
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
    }


def metrics(y, pred, threshold=0.25, errors=False):
    y, pred = np.asarray(y), np.asarray(pred)
    result = {"n": len(y), **correlations(y, pred)}
    mask = (np.abs(y) >= threshold) & (y != 0)
    truth, guess = np.sign(y[mask]), np.sign(pred[mask])
    result.update(
        {
            "n_sign": int(mask.sum()),
            "sign_accuracy": float(np.mean(truth == guess)) if mask.any() else None,
            "sign_coverage": float(np.mean(guess != 0)) if mask.any() else None,
        }
    )
    result["balanced_sign_accuracy"] = (
        float(np.mean([np.mean(guess[truth == c] == c) for c in (-1, 1)]))
        if set(truth) == {-1, 1}
        else None
    )
    if errors:
        result.update(
            rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
            mae=float(np.mean(np.abs(y - pred))),
        )
    return result


def noise_ceiling(y, epsilon_se, significance_z=1.96):
    """Estimate target reliability and the corresponding observed-r ceiling.

    This is the classical independent-error approximation
    ``R = 1 - mean(epsilon_se**2) / Var(epsilon)``.  The square-root of R is
    the expected maximum observed correlation for a perfect predictor.  The
    estimate is a diagnostic: it assumes the reported standard errors capture
    independent measurement error and that the across-pair variance is the
    signal variance plus that error variance.
    """
    y, epsilon_se = np.asarray(y, dtype=float), np.asarray(epsilon_se, dtype=float)
    if y.shape != epsilon_se.shape:
        raise ValueError("y and epsilon_se must have the same shape")
    mask = np.isfinite(y) & np.isfinite(epsilon_se) & (epsilon_se >= 0)
    if mask.sum() < 2:
        raise ValueError("Need at least two finite observations with nonnegative epsilon_se")
    y, epsilon_se = y[mask], epsilon_se[mask]
    epsilon_variance = float(np.var(y, ddof=0))
    mean_se_squared = float(np.mean(epsilon_se**2))
    if epsilon_variance <= 0:
        reliability_raw = None
        reliability = 0.0
    else:
        reliability_raw = float(1.0 - mean_se_squared / epsilon_variance)
        reliability = float(np.clip(reliability_raw, 0.0, 1.0))
    return {
        "n": int(mask.sum()),
        "epsilon_variance_population": epsilon_variance,
        "mean_epsilon_se_squared": mean_se_squared,
        "estimated_signal_variance": float(max(epsilon_variance - mean_se_squared, 0.0)),
        "reliability_raw": reliability_raw,
        "reliability": reliability,
        "perfect_predictor_observed_correlation_ceiling": float(np.sqrt(reliability)),
        "distinguishable_n_abs_epsilon_ge_z_se": int(
            np.sum(np.abs(y) >= significance_z * epsilon_se)
        ),
        "distinguishable_fraction_abs_epsilon_ge_z_se": float(
            np.mean(np.abs(y) >= significance_z * epsilon_se)
        ),
        "significance_z": float(significance_z),
        "assumption": "classical independent measurement error; diagnostic estimate",
    }


def group_boot(groups, statistic, n_boot=1000, seed=0):
    """Percentile interval resampling whole overlap groups."""
    rng = np.random.default_rng(seed)
    unique = np.asarray(pd.unique(groups))
    index = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        value = statistic(np.concatenate([index[g] for g in picked]))
        if np.isfinite(value):
            values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ranks = pd.Series(score).rank().to_numpy()
    positives, negatives = label.sum(), (~label).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def gc_fraction(sequences):
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in sequences])


def element_kmers(sequences, k_max=3):
    """Count overlapping DNA words in length-major, A/C/G/T order."""
    vocab = (
        KMER_VOCAB
        if k_max == 3
        else [
            "".join(word) for size in range(1, k_max + 1) for word in product("ACGT", repeat=size)
        ]
    )
    lookup = {k: i for i, k in enumerate(vocab)}
    features = np.zeros((len(sequences), len(vocab)))
    for row, sequence in enumerate(sequences):
        for size in range(1, k_max + 1):
            for start in range(len(sequence) - size + 1):
                column = lookup.get(sequence[start : start + size])
                if column is not None:
                    features[row, column] += 1
    return features


def rank_feature_contrasts(activations, top_k=20):
    """Rank SAE features; rows must be WT,A,B,AB and use the benchmark sign."""
    x = np.asarray(activations)
    if x.ndim != 2 or x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Expected finite activations with shape (4, features)")
    contrast = x[1] + x[2] - x[0] - x[3]
    order = np.argsort(-np.abs(contrast), kind="stable")[:top_k]
    return pd.DataFrame(
        {
            "feature": order,
            "contrast": contrast[order],
            "abs_contrast": np.abs(contrast[order]),
        }
    )
