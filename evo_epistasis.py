#!/usr/bin/env python3
"""Minimal Evo 2 benchmark for four-haplotype regulatory epistasis."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATES = ("wt", "a", "b", "ab")
SOURCE = "https://zenodo.org/records/15297965"
CONTRAST = "A+B-WT-AB (expected additive minus observed double)"
KMER_VOCAB = tuple("".join(chars) for k in (1, 2, 3) for chars in itertools.product("ACGT", repeat=k))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seq_id(seq):
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def differences(ref, seq):
    if len(ref) != len(seq):
        raise ValueError("Only equal-length substitutions are supported")
    return [(i, r, a) for i, (r, a) in enumerate(zip(ref, seq)) if r != a]


def mutate(ref, changes):
    out = list(ref)
    for i, old, new in changes:
        if ref[i] != old:
            raise ValueError("Reference allele mismatch")
        out[i] = new
    return "".join(out)


def load_quartets(path):
    """Load and strictly validate one K562 row per WT/A/B/AB quartet."""
    df = pd.read_csv(path)
    required = {"pair_id", "group_id", "condition", "pos_a", "pos_b", "distance", "epsilon"}
    required |= {f"{kind}_{state}" for kind in ("seq", "id", "y") for state in STATES}
    if missing := required - set(df):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df.pair_id.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError("Quartets must be nonempty, unique, and complete")
    if df.condition.nunique() != 1:
        raise ValueError("Evaluate one experimental condition at a time")
    for state in STATES:
        seq = df[f"seq_{state}"].astype(str)
        if not seq.str.fullmatch("[ACGT]+", na=False).all():
            raise ValueError("Sequences must be uppercase A/C/G/T")
        if not all(seq_id(s) == i for s, i in zip(seq, df[f"id_{state}"])):
            raise ValueError("Sequence hash mismatch")
        df[f"y_{state}"] = pd.to_numeric(df[f"y_{state}"], errors="raise")
    numeric = [f"y_{s}" for s in STATES] + ["epsilon", "distance", "pos_a", "pos_b"]
    if not np.isfinite(df[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite measurement or coordinate")
    if "epsilon_se" in df:
        df["epsilon_se"] = pd.to_numeric(df["epsilon_se"], errors="raise")
        if not np.isfinite(df.epsilon_se.to_numpy(float)).all() or (df.epsilon_se < 0).any():
            raise ValueError("epsilon_se must be finite and nonnegative")
    expected = df.y_a + df.y_b - df.y_wt - df.y_ab
    if not np.allclose(df.epsilon, expected, atol=1e-8, rtol=1e-8):
        raise ValueError(f"epsilon must use {CONTRAST}")
    for row in df.itertuples():
        da, db, dab = differences(row.seq_wt, row.seq_a), differences(row.seq_wt, row.seq_b), differences(row.seq_wt, row.seq_ab)
        if len(da) != 1 or len(db) != 1 or sorted(da + db) != dab:
            raise ValueError(f"{row.pair_id}: sequences are not a complete two-SNV quartet")
        if (da[0][0] + 1, db[0][0] + 1, db[0][0] - da[0][0]) != (row.pos_a, row.pos_b, row.distance):
            raise ValueError(f"{row.pair_id}: mutation coordinates do not match sequences")
    return df


def parse_variant(value):
    value = value.replace("chr:", "chr")
    match = re.fullmatch(r"(chr[0-9XY]+):(\d+):([ACGT]):([ACGT])", value)
    if not match or match.group(3) == match.group(4):
        raise ValueError("not_biallelic_SNV")
    chrom, pos, ref, alt = match.groups()
    return value, chrom, int(pos), ref, alt


def read_oligos(code_zip, wanted):
    found = {}
    def add(key, value):
        if key in found and found[key] != value:
            raise ValueError(f"Conflicting FASTA records for {key}")
        found[key] = value
    with zipfile.ZipFile(code_zip) as archive:
        for name in archive.namelist():
            if "/dockerfiles/mpra_chr" not in name or not name.endswith(".fasta"):
                continue
            key, parts = None, []
            with archive.open(name) as handle:
                for raw in handle:
                    line = raw.decode().strip()
                    if line.startswith(">"):
                        if key in wanted:
                            add(key, "".join(parts))
                        key, parts = line[1:].split()[0], []
                    elif key in wanted:
                        parts.append(line)
                if key in wanted:
                    add(key, "".join(parts))
    return found


def reconstruct(row, fasta):
    v1, chrom1, p1, r1, a1 = parse_variant(row.v1)
    v2, chrom2, p2, r2, a2 = parse_variant(row.v2)
    if chrom1 != chrom2 or p1 == p2:
        raise ValueError("different_chromosome_or_same_site")
    ref, single1 = fasta.get(v1 + "_allele1_oligo"), fasta.get(v1 + "_allele2_oligo")
    ref2, alt2 = fasta.get(v2 + "_allele1_oligo"), fasta.get(v2 + "_allele2_oligo")
    if None in (ref, single1, ref2, alt2):
        raise ValueError("missing_oligo")
    if any(len(s) != 200 or not re.fullmatch("[ACGT]+", s) for s in (ref, single1, ref2, alt2)):
        raise ValueError("not_200nt_unambiguous_DNA")
    if differences(ref, single1) != [(99, r1, a1)] or differences(ref2, alt2) != [(99, r2, a2)]:
        raise ValueError("central_allele_or_coordinate_mismatch")
    index2 = 99 + p2 - p1
    if not 10 <= index2 < 190 or ref[index2] != r2:
        raise ValueError("second_variant_outside_window_or_reference_mismatch")
    start1, start2 = p1 - 99, p2 - 99
    lo, hi = max(start1, start2), min(start1 + 200, start2 + 200)
    if ref[lo-start1:hi-start1] != ref2[lo-start2:hi-start2]:
        raise ValueError("reference_oligo_overlap_mismatch")
    single2 = mutate(ref, [(index2, r2, a2)])
    double = mutate(single1, [(index2, r2, a2)])
    return [ref, single1, single2, double], chrom1, start1, start1 + 199


def prepare_siraj(windows_path, code_zip, out, cell="K562", min_dna=20.0, max_se=0.5):
    """Reconstruct the fixed 200-nt K562 middle-window benchmark and its audit trail."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(windows_path, sep="\t")
    source_rows = len(source)
    source = source[(source.cell_type == cell) & (source.window == "middle") & (source.center_variant == "var1")].copy()
    source[["v1", "v2"]] = source[["v1", "v2"]].replace("chr:", "chr", regex=True)
    source = source.sort_values(["v1", "v2", "library", "v1v2_construct"])
    if source.duplicated(["v1", "v2", "library"]).any():
        raise ValueError("Unexpected duplicate pair within a library")
    duplicate = source.duplicated(["v1", "v2"])
    audit = [dict(status="not_selected", reason="lexicographic_library_rule", **r) for r in source[duplicate].to_dict("records")]
    source = source[~duplicate].copy()
    wanted = {v + suffix for v in set(source.v1) | set(source.v2) for suffix in ("_allele1_oligo", "_allele2_oligo")}
    fasta, accepted = read_oligos(code_zip, wanted), []
    for row in source.itertuples(index=False):
        pair_id = f"{row.v1};{row.v2};var1;middle;{row.library}"
        try:
            seqs, chrom, start, end = reconstruct(row, fasta)
            names = ("refref", "altref", "refalt", "altalt")
            y = np.array([0.0, row.altref_log2Skew, row.refalt_log2Skew, row.altalt_log2Skew], float)
            se = np.array([getattr(row, f"{x}_Log2FC_SE") for x in names], float)
            dna = np.array([getattr(row, f"mean_Plasmid_{x}") for x in names], float)
            if not np.isfinite(np.r_[y, se, dna]).all() or (se < 0).any() or (dna < 0).any():
                raise ValueError("nonfinite_or_invalid_measurement")
            if (dna < min_dna).any():
                raise ValueError("mean_DNA_below_threshold")
            if (se > max_se).any():
                raise ValueError("activity_SE_above_threshold")
            epsilon = y[1] + y[2] - y[0] - y[3]
            if not np.isfinite(row.int_log2Skew) or not np.isclose(epsilon, -row.int_log2Skew, atol=1e-6, rtol=1e-6):
                raise ValueError("interaction_contrast_mismatch")
            accepted.append((pair_id, seqs, y, row.int_log2SkewSE, chrom, start, end))
            audit.append(dict(status="accepted", reason="", **row._asdict()))
        except ValueError as error:
            audit.append(dict(status="excluded", reason=str(error), pair_id=pair_id, **row._asdict()))
    rows, active_chrom, active_end, group = [], None, -1, -1
    for pair_id, seqs, y, epsilon_se, chrom, start, end in sorted(accepted, key=lambda x: (x[4], x[5], x[6])):
        if chrom != active_chrom or start > active_end:
            group += 1; active_chrom, active_end = chrom, end
        active_end = max(active_end, end)
        singles = sorted([(differences(seqs[0], seqs[i])[0][0], seqs[i], y[i]) for i in (1, 2)])
        ordered_seq = [seqs[0], singles[0][1], singles[1][1], seqs[3]]
        ordered_y = [y[0], singles[0][2], singles[1][2], y[3]]
        pos_a, pos_b = singles[0][0] + 1, singles[1][0] + 1
        row = {"pair_id": f"{pair_id}|{cell}|{seq_id(ordered_seq[3])[:16]}", "background_id": pair_id,
               "condition": cell, "group_id": f"{cell}_region_{group:05d}",
               "pos_a": pos_a, "pos_b": pos_b, "distance": pos_b - pos_a}
        for state, seq, value in zip(STATES, ordered_seq, ordered_y):
            row.update({f"seq_{state}": seq, f"id_{state}": seq_id(seq), f"y_{state}": value})
        row.update(epsilon=ordered_y[1] + ordered_y[2] - ordered_y[0] - ordered_y[3], epsilon_se=epsilon_se)
        rows.append(row)
    quartets = pd.DataFrame(rows)
    if quartets.empty:
        raise ValueError("No accepted quartets")
    qpath, apath = out / "quartets.csv.gz", out / "audit.csv.gz"
    quartets.to_csv(qpath, index=False); pd.DataFrame(audit).to_csv(apath, index=False)
    load_quartets(qpath)
    report = {"source": SOURCE, "source_rows": source_rows, "cell": cell, "window": "middle", "center_variant": "var1",
              "contrast": CONTRAST, "source_int_log2Skew_relation": "epsilon = -int_log2Skew",
              "library_rule": "lexicographic_first_before_QC", "candidate_pairs": len(source),
              "duplicate_libraries_not_selected": int(duplicate.sum()), "accepted_pairs": len(quartets),
              "excluded_pairs": int(sum(x["status"] == "excluded" for x in audit)),
              "overlap_groups": int(quartets.group_id.nunique()), "min_mean_DNA": min_dna,
              "max_activity_SE_log2": max_se, "all_windows_sha256": file_hash(windows_path),
              "code_zip_sha256": file_hash(code_zip), "original_haplotype_design_independently_verified": False}
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class EvoScorer:
    def __init__(self, checkpoint, weights=None, device="cuda:0"):
        import torch
        from evo2 import Evo2
        if not torch.cuda.is_available():
            raise RuntimeError("Evo scoring requires a supported NVIDIA GPU")
        self.torch, self.model, self.device = torch, Evo2(checkpoint, local_path=weights), device
        self.model.model.eval()

    def forward(self, sequences):
        torch, tokenizer = self.torch, self.model.tokenizer
        tokens = [tokenizer.tokenize(s) for s in sequences]
        if len({len(s) for s in sequences}) != 1 or any(len(t) != len(s) for t, s in zip(tokens, sequences)):
            raise ValueError("Scoring requires equal lengths and one token per nucleotide")
        ids = torch.tensor([[tokenizer.eod_id] + t for t in tokens], dtype=torch.long, device=self.device)
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


def score_quartets(quartets_path, output_path, backend="evo", checkpoint="evo2_7b_base",
                   revision="UNRECORDED", batch_size=1, weights=None):
    """Score each unique sequence once; the output CSV is also the resumable cache."""
    if backend not in {"evo", "gc"} or batch_size < 1:
        raise ValueError("backend must be evo or gc, and batch_size must be positive")
    if backend == "evo" and revision == "UNRECORDED":
        raise ValueError("Record the exact checkpoint snapshot with --revision")
    quartets = load_quartets(quartets_path)
    sequences = pd.concat([quartets[[f"id_{s}", f"seq_{s}"]].set_axis(["sequence_id", "sequence"], axis=1) for s in STATES])
    sequences = sequences.drop_duplicates().sort_values("sequence_id")
    if sequences.sequence_id.duplicated().any():
        raise ValueError("One sequence hash maps to multiple sequences")
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.stem + ".meta.json")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}
    score_definition = ("mean(forward,RC) of summed AR log-likelihood; EOD-as-BOS; "
                        "FP32 log-softmax; FP64 sum" if backend == "evo" else "GC count; additive negative control")
    meta = {"backend": backend, "checkpoint": checkpoint if backend == "evo" else None,
            "revision": revision if backend == "evo" else None, "contrast": CONTRAST,
            "score": score_definition,
            "quartets_sha256": file_hash(quartets_path), "code_sha256": file_hash(__file__),
            "weights_sha256": file_hash(weights) if weights else None, "batch_size": batch_size,
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions}
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Scoring configuration changed; use a new output path")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    cached = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=["sequence_id", "forward", "reverse", "score"])
    if cached.sequence_id.duplicated().any() or (len(cached) and not np.isfinite(cached[["forward", "reverse", "score"]]).all().all()):
        raise ValueError("Invalid score cache")
    pending = sequences[~sequences.sequence_id.isin(cached.sequence_id)]
    scorer = EvoScorer(checkpoint, weights) if backend == "evo" and len(pending) else None
    for _, length_group in pending.groupby(pending.sequence.str.len(), sort=True):
        for start in range(0, len(length_group), batch_size):
            batch = length_group.iloc[start:start + batch_size]
            seqs = batch.sequence.tolist()
            if scorer:
                forward, reverse = scorer.forward(seqs), scorer.forward([reverse_complement(s) for s in seqs])
            else:
                forward = reverse = np.array([s.count("G") + s.count("C") for s in seqs], float)
            rows = pd.DataFrame({"sequence_id": batch.sequence_id, "forward": forward, "reverse": reverse,
                                 "score": (np.asarray(forward) + np.asarray(reverse)) / 2})
            rows.to_csv(output, mode="a", header=not output.exists(), index=False)
    scores = pd.read_csv(output).drop_duplicates("sequence_id").set_index("sequence_id").loc[sequences.sequence_id].reset_index()
    scores.to_csv(output, index=False)
    return {"sequences": len(scores), "newly_scored": len(pending), **meta}


def correlations(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) < 3 or np.ptp(y) < 1e-12 or np.ptp(pred) < 1e-12:
        return {"spearman": None, "pearson": None}
    return {"spearman": float(spearmanr(y, pred).statistic), "pearson": float(pearsonr(y, pred).statistic)}


def metrics(y, pred, threshold=0.25, errors=False):
    y, pred = np.asarray(y), np.asarray(pred)
    result = {"n": len(y), **correlations(y, pred)}
    mask = (np.abs(y) >= threshold) & (y != 0)
    truth, guess = np.sign(y[mask]), np.sign(pred[mask])
    result.update({"n_sign": int(mask.sum()), "sign_accuracy": float(np.mean(truth == guess)) if mask.any() else None,
                   "sign_coverage": float(np.mean(guess != 0)) if mask.any() else None})
    result["balanced_sign_accuracy"] = float(np.mean([np.mean(guess[truth == c] == c) for c in (-1, 1)])) if set(truth) == {-1, 1} else None
    if errors:
        result.update(rmse=float(np.sqrt(np.mean((y - pred) ** 2))), mae=float(np.mean(np.abs(y - pred))))
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
    mean_se_squared = float(np.mean(epsilon_se ** 2))
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
        "distinguishable_n_abs_epsilon_ge_z_se": int(np.sum(np.abs(y) >= significance_z * epsilon_se)),
        "distinguishable_fraction_abs_epsilon_ge_z_se": float(np.mean(np.abs(y) >= significance_z * epsilon_se)),
        "significance_z": float(significance_z),
        "assumption": "classical independent measurement error; diagnostic estimate",
    }


def sequence_only_features(quartets):
    """Build sequence/design features without using measured activity values."""
    features = np.zeros((len(quartets), len(STATES) * len(KMER_VOCAB) + 3), float)
    for row_index, row in enumerate(quartets.itertuples(index=False)):
        offset = 0
        for state in STATES:
            sequence = getattr(row, f"seq_{state}")
            for kmer_index, kmer in enumerate(KMER_VOCAB):
                features[row_index, offset + kmer_index] = sum(
                    sequence[i:i + len(kmer)] == kmer for i in range(len(sequence) - len(kmer) + 1)
                )
            offset += len(KMER_VOCAB)
        features[row_index, -3:] = (row.pos_a, row.pos_b, row.distance)
    return features


def add_predictions(quartets, scores, folds=5, seed=0):
    if scores.sequence_id.duplicated().any():
        raise ValueError("Duplicate sequence scores")
    df, mapping = quartets.copy(), scores.set_index("sequence_id").score
    for state in STATES:
        df[f"s_{state}"] = df[f"id_{state}"].map(mapping)
    if not np.isfinite(df[[f"s_{s}" for s in STATES]].to_numpy()).all():
        raise ValueError("Missing or nonfinite sequence score")
    df["model_interaction"] = df.s_a + df.s_b - df.s_wt - df.s_ab
    if not 2 <= folds <= df.group_id.nunique():
        raise ValueError("Invalid grouped-fold count")
    y = df.epsilon.to_numpy()
    features = sequence_only_features(df)
    df["additive_zero"], df["fold"] = 0.0, -1
    names = ("calibrated_model", "training_mean", "training_median", "sequence_only_kmer_ridge", "majority_sign")
    for name in names:
        df[name] = np.nan
    coefficients = []
    split = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(split.split(df, groups=df.group_id)):
        model = LinearRegression().fit(df.model_interaction.to_numpy()[train, None], y[train])
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features[train], y[train])
        df.loc[df.index[test], "calibrated_model"] = model.predict(df.model_interaction.to_numpy()[test, None])
        df.loc[df.index[test], "sequence_only_kmer_ridge"] = ridge.predict(features[test])
        df.loc[df.index[test], "training_mean"] = y[train].mean()
        df.loc[df.index[test], "training_median"] = np.median(y[train])
        df.loc[df.index[test], "majority_sign"] = 1.0 if np.sum(y[train] > 0) >= np.sum(y[train] < 0) else -1.0
        df.loc[df.index[test], "fold"] = fold
        coefficients.append({"fold": fold, "intercept": float(model.intercept_), "slope": float(model.coef_[0])})
    return df, coefficients


def cluster_intervals(df, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(df.group_id.to_numpy() == g) for g in df.group_id.unique()]
    samples = {k: [] for k in ("spearman", "pearson", "rmse_gain_zero", "rmse_gain_mean", "rmse_gain_sequence")}
    y, raw, calibrated = (df[c].to_numpy() for c in ("epsilon", "model_interaction", "calibrated_model"))
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        for key, value in correlations(y[idx], raw[idx]).items():
            if value is not None:
                samples[key].append(value)
        model_rmse = np.sqrt(np.mean((y[idx] - calibrated[idx]) ** 2))
        for key, column in (("rmse_gain_zero", "additive_zero"), ("rmse_gain_mean", "training_mean"),
                            ("rmse_gain_sequence", "sequence_only_kmer_ridge")):
            samples[key].append(float(np.sqrt(np.mean((y[idx] - df[column].to_numpy()[idx]) ** 2)) - model_rmse))
    return {k: np.quantile(v, [0.025, 0.975]).tolist() if v else None for k, v in samples.items()}


def select_cases(df, n=3, threshold=0.25):
    eligible = df[df.epsilon.abs() >= threshold].copy()
    eligible["correct_sign"] = np.sign(eligible.epsilon) == np.sign(eligible.model_interaction)
    eligible["oof_abs_error"] = (eligible.epsilon - eligible.calibrated_model).abs()
    success = eligible[eligible.correct_sign].sort_values(["oof_abs_error", "pair_id"]).head(n).assign(case="success")
    failure = eligible[~eligible.correct_sign].sort_values(["epsilon", "pair_id"], key=lambda x: -x.abs() if x.name == "epsilon" else x).head(n).assign(case="failure")
    control = df[df.epsilon.abs() < threshold].sort_values("epsilon", key=lambda x: x.abs()).head(n).assign(case="near_additive")
    return pd.concat([success, failure, control], ignore_index=True)


def rank_feature_contrasts(activations, top_k=20):
    """Rank SAE features; rows must be WT,A,B,AB and use the benchmark sign."""
    x = np.asarray(activations)
    if x.ndim != 2 or x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Expected finite activations with shape (4, features)")
    contrast = x[1] + x[2] - x[0] - x[3]
    order = np.argsort(-np.abs(contrast), kind="stable")[:top_k]
    return pd.DataFrame({"feature": order, "contrast": contrast[order], "abs_contrast": np.abs(contrast[order])})


def make_plot(df, path, label):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].scatter(df.model_interaction, df.epsilon, s=8, alpha=0.4)
    axes[0, 0].set(xlabel=f"{label} interaction", ylabel="Measured epistasis", title="Raw association")
    axes[0, 1].scatter(df.calibrated_model, df.epsilon, s=8, alpha=0.4)
    lo, hi = min(df.calibrated_model.min(), df.epsilon.min()), max(df.calibrated_model.max(), df.epsilon.max())
    axes[0, 1].plot([lo, hi], [lo, hi], color="grey"); axes[0, 1].set(xlabel="OOF prediction", ylabel="Measured epistasis", title="Calibrated")
    axes[0, 2].hist(df.epsilon, bins=40); axes[0, 2].set(title="Measured epistasis", xlabel="log2 activity")
    axes[1, 0].hist(df.model_interaction, bins=40); axes[1, 0].set(title="Model interaction", xlabel="score units")
    bins = (("distance", pd.cut(df.distance, [0, 10, 25, 50, 100, np.inf])),
            ("|epistasis|", pd.cut(df.epsilon.abs(), [0, 0.25, 0.5, 1, np.inf], include_lowest=True)))
    strata = []
    for ax, (name, groups) in zip(axes[1, 1:], bins):
        values = df.assign(bin=groups).groupby("bin", observed=True).apply(
            lambda x: pd.Series({"n": len(x), "mae": np.mean(np.abs(x.epsilon - x.calibrated_model)),
                                 **correlations(x.epsilon, x.model_interaction)}), include_groups=False)
        ax.bar(values.index.astype(str), values.mae); ax.tick_params(axis="x", rotation=25)
        ax.set(title=f"Error by {name}", ylabel="OOF MAE")
        strata.extend({"stratification": name, "bin": str(i), "n": int(r.n), "mae": float(r.mae),
                       "spearman": None if pd.isna(r.spearman) else float(r.spearman),
                       "pearson": None if pd.isna(r.pearson) else float(r.pearson)} for i, r in values.iterrows())
    fig.suptitle(f"{label}: {CONTRAST}"); fig.savefig(path, dpi=180); plt.close(fig)
    return strata


def evaluate(quartets_path, scores_path, out, label="Evo 2", folds=5, seed=0, n_boot=1000, threshold=0.25):
    if n_boot < 1 or threshold < 0:
        raise ValueError("bootstrap must be positive and threshold nonnegative")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df, coefficients = add_predictions(load_quartets(quartets_path), pd.read_csv(scores_path), folds, seed)
    df.to_csv(out / "predictions.csv", index=False)
    select_cases(df, threshold=threshold).to_csv(out / "cases.csv", index=False)
    strata = make_plot(df, out / "plots.png", label)
    y = df.epsilon.to_numpy()
    prediction_metrics = {x: metrics(y, df[x], threshold, True) for x in
                          ("calibrated_model", "additive_zero", "training_mean", "training_median", "sequence_only_kmer_ridge")}
    table = pd.DataFrame([{"method": "raw " + label, **metrics(y, df.model_interaction, threshold)},
                          *[{"method": name, **values} for name, values in prediction_metrics.items()]])
    table.to_csv(out / "results_table.csv", index=False)
    result = {"label": label, "contrast": CONTRAST, "pairs": len(df), "groups": int(df.group_id.nunique()),
              "raw": metrics(y, df.model_interaction, threshold),
              "raw_all_nonzero_signs": metrics(y, df.model_interaction, 0),
              "predictions": prediction_metrics,
              "majority_sign": metrics(y, df.majority_sign, threshold), "calibration": coefficients,
              "cluster_bootstrap_95ci": cluster_intervals(df, n_boot, seed), "strata": strata,
              "settings": {"folds": folds, "seed": seed, "bootstrap": n_boot, "sign_threshold": threshold},
              "inputs": {"quartets_sha256": file_hash(quartets_path), "scores_sha256": file_hash(scores_path)}}
    if "epsilon_se" in df:
        se = pd.to_numeric(df.epsilon_se, errors="coerce")
        result["noise_ceiling"] = noise_ceiling(y, se)
        reliability = result["noise_ceiling"]["reliability"]
        if reliability > 0:
            scale = np.sqrt(reliability)
            ci = result["cluster_bootstrap_95ci"]
            result["noise_ceiling"].update({
                "spearman_upper_95_reliability_adjusted_approx": float(np.clip(ci["spearman"][1] / scale, -1.0, 1.0)),
                "pearson_upper_95_reliability_adjusted": float(np.clip(ci["pearson"][1] / scale, -1.0, 1.0)),
                "correlation_adjustment_note": "Spearman adjustment is approximate; classical attenuation correction is defined for Pearson correlation.",
            })
        mask = np.isfinite(se) & (se > 0) & (df.epsilon.abs() > 1.96 * se)
        result["exploratory_source_SE_subset"] = metrics(df.epsilon[mask], df.model_interaction[mask], threshold)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-siraj")
    p.add_argument("--windows", required=True); p.add_argument("--code-zip", required=True); p.add_argument("--out", required=True)
    p.add_argument("--cell", default="K562"); p.add_argument("--min-dna", type=float, default=20); p.add_argument("--max-se", type=float, default=0.5)
    p = commands.add_parser("score")
    p.add_argument("--quartets", required=True); p.add_argument("--output", required=True); p.add_argument("--backend", choices=("evo", "gc"), default="evo")
    p.add_argument("--checkpoint", default="evo2_7b_base"); p.add_argument("--revision", default="UNRECORDED")
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--weights")
    p = commands.add_parser("evaluate")
    p.add_argument("--quartets", required=True); p.add_argument("--scores", required=True); p.add_argument("--out", required=True)
    p.add_argument("--label", default="Evo 2"); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000); p.add_argument("--sign-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "prepare-siraj":
        result = prepare_siraj(args.windows, args.code_zip, args.out, args.cell, args.min_dna, args.max_se)
    elif args.command == "score":
        result = score_quartets(args.quartets, args.output, args.backend, args.checkpoint, args.revision, args.batch_size, args.weights)
    else:
        result = evaluate(args.quartets, args.scores, args.out, args.label, args.folds, args.seed, args.bootstrap, args.sign_threshold)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
