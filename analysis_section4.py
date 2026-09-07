#!/usr/bin/env python3
"""Reproduce every number in the write-up's results section from shipped outputs.

Reads results/evo2_7b_base/predictions.csv and data/audit.csv.gz only.
No GPU, no downloads, no re-scoring. Run from the repo root:

    python3 analysis_section4.py

Writes results/section4/numbers.json and prints a report.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATES = ("wt", "a", "b", "ab")
KMER_VOCAB = tuple("".join(c) for k in (1, 2, 3) for c in itertools.product("ACGT", repeat=k))
RNG_SEED = 0
N_BOOT = 1000


# --------------------------------------------------------------------------- io

def load(repo):
    pred = pd.read_csv(repo / "results/evo2_7b_base/predictions.csv")
    audit = pd.read_csv(repo / "data/audit.csv.gz")
    key = ["v1", "v2", "center_variant", "window", "library"]
    missing = [c for c in key if c not in audit.columns]
    if missing:
        raise SystemExit(f"audit.csv.gz is missing join columns: {missing}")
    joined = audit[key[0]].astype(str)
    for col in key[1:]:
        joined = joined + ";" + audit[col].astype(str)
    audit = audit.assign(_k=joined)
    pred = pred.assign(_k=pred["pair_id"].str.split("|").str[0])
    keep = ["_k", "refref_Log2FC", "refref_active", "int_emVar", "int_log2Skew", "int_log2SkewSE"]
    keep = [c for c in keep if c in audit.columns]
    df = pred.merge(audit.drop_duplicates("_k")[keep], on="_k", how="left")
    if len(df) != len(pred):
        raise SystemExit("join changed row count")
    matched = df["int_emVar"].notna().sum() if "int_emVar" in df else 0
    print(f"join: {matched}/{len(df)} rows matched to audit")
    return df


# ------------------------------------------------------------------- recoding

def add_flip(df):
    """Paper's recoding: lowest-activity diplotype becomes the reference category.

    The four diplotypes pair into complements (wt<->ab, a<->b), so this can only
    flip the sign of the second difference. Magnitude never changes.
    """
    y = df[[f"y_{s}" for s in STATES]].to_numpy(float)
    lowest = np.argmin(y, axis=1)
    flip = np.where(np.isin(lowest, [0, 3]), 1.0, -1.0)
    return df.assign(
        flip=flip,
        epsilon_refalt=df["epsilon"],
        epsilon_recoded=flip * df["epsilon"],
        model_recoded=flip * df["model_interaction"],
    )


def reproduction(df):
    lab = df[df["int_emVar"] == True]  # noqa: E712
    out = {
        "n_pairs": int(len(df)),
        "n_labelled": int(len(lab)),
        "flip_negative": int((df["flip"] < 0).sum()),
        "damp_frac_labelled_refalt": float((lab["epsilon_refalt"] > 0).mean()),
        "damp_frac_labelled_recoded": float((lab["epsilon_recoded"] > 0).mean()),
        "damp_frac_all_recoded": float((df["epsilon_recoded"] > 0).mean()),
        "paper_damp_frac": 139 / 180,
    }
    print("\n== 1. recoding / reproduction ==")
    print(f"  flip = -1 on {out['flip_negative']} of {out['n_pairs']} pairs")
    print(f"  labelled non-additive pairs: n = {out['n_labelled']}")
    print(f"  dampening among labelled, ref/alt coding : {out['damp_frac_labelled_refalt']:.3f}")
    print(f"  dampening among labelled, recoded        : {out['damp_frac_labelled_recoded']:.3f}")
    print(f"  dampening over all pairs, recoded        : {out['damp_frac_all_recoded']:.3f}")
    print(f"  paper, 139/180 called non-additive pairs  : {out['paper_damp_frac']:.3f}")
    return out


# -------------------------------------------------------------- bootstrap util

def group_boot(groups, stat, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile CI resampling whole overlap groups."""
    rng = np.random.default_rng(seed)
    uniq = np.asarray(pd.unique(groups))
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------- single variants

def singles(df):
    dy = np.concatenate([df["y_a"].to_numpy(float), df["y_b"].to_numpy(float)])
    ds = np.concatenate([
        (df["s_a"] - df["s_wt"]).to_numpy(float),
        (df["s_b"] - df["s_wt"]).to_numpy(float),
    ])
    grp = np.concatenate([df["group_id"].to_numpy(), df["group_id"].to_numpy()])
    mag = spearmanr(np.abs(ds), np.abs(dy)).statistic
    signed = spearmanr(ds, dy).statistic
    lo, hi = group_boot(grp, lambda i: spearmanr(np.abs(ds[i]), np.abs(dy[i])).statistic)
    out = {"n": int(len(dy)), "magnitude_spearman": float(mag), "signed_spearman": float(signed),
           "magnitude_ci": [lo, hi], "sd_single_effect": float(np.std(ds, ddof=1)),
           "sd_s_wt": float(df["s_wt"].std())}
    print("\n== 2. single variants, within element ==")
    print(f"  n = {out['n']}")
    print(f"  magnitude Spearman |dEvo| vs |dActivity| : {mag:+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  signed Spearman                          : {signed:+.4f}")
    print(f"  join check: SD(Evo single-variant score change) {out['sd_single_effect']:.2f} vs SD(s_wt) "
          f"{out['sd_s_wt']:.2f}; a scrambled join would give ~{out['sd_s_wt']*np.sqrt(2):.0f}")
    return out


# -------------------------------------------------------------------- elements

def gc_frac(seqs):
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])


def kmer_counts(seqs):
    idx = {k: i for i, k in enumerate(KMER_VOCAB)}
    X = np.zeros((len(seqs), len(KMER_VOCAB)))
    for r, s in enumerate(seqs):
        for k in (1, 2, 3):  # overlapping windows; str.count would undercount
            for p in range(len(s) - k + 1):
                j = idx.get(s[p:p + k])
                if j is not None:
                    X[r, j] += 1
    return X


def oof_linear(X, y, groups, folds=5, ridge=False):
    pred = np.empty(len(y))
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    for train, test in GroupKFold(n_splits=folds).split(X, y, groups):
        model = (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30)))
                 if ridge else make_pipeline(StandardScaler(), LinearRegression()))
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred


def elements(df):
    el = (df.dropna(subset=["refref_Log2FC"])
            .drop_duplicates("seq_wt")[["seq_wt", "s_wt", "refref_Log2FC", "refref_active", "group_id"]]
            .reset_index(drop=True))
    spans = el.groupby("seq_wt")["group_id"].nunique()
    if (spans > 1).any():
        raise SystemExit("a reference sequence spans more than one CV group")
    y = el["refref_Log2FC"].to_numpy(float)
    gc = gc_frac(el["seq_wt"])
    evo = el["s_wt"].to_numpy(float)
    grp = el["group_id"].to_numpy()
    active = el["refref_active"].astype(str).str.lower().isin(["true", "1"]).to_numpy()

    km = kmer_counts(el["seq_wt"])
    p_gc = oof_linear(gc, y, grp)
    p_evo = oof_linear(evo, y, grp)
    p_km = oof_linear(km, y, grp, ridge=True)
    mean_pred = np.empty(len(y))
    for train, test in GroupKFold(n_splits=5).split(y.reshape(-1, 1), y, grp):
        mean_pred[test] = y[train].mean()

    def rmse(p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    out = {
        "n_elements": int(len(el)),
        "raw": {
            "gc_vs_activity": float(spearmanr(gc, y).statistic),
            "evo_vs_activity": float(spearmanr(evo, y).statistic),
            "evo_vs_activity_active_only": float(spearmanr(evo[active], y[active]).statistic),
            "n_active": int(active.sum()),
            "evo_vs_gc": float(spearmanr(evo, gc).statistic),
        },
        "oof": {
            "gc": [float(spearmanr(p_gc, y).statistic), rmse(p_gc)],
            "evo": [float(spearmanr(p_evo, y).statistic), rmse(p_evo)],
            "kmer_1_2_3": [float(spearmanr(p_km, y).statistic), rmse(p_km)],
            "training_mean_rmse": rmse(mean_pred),
        },
    }
    lo, hi = group_boot(grp, lambda i: spearmanr(gc[i], y[i]).statistic)
    out["raw"]["gc_ci"] = [lo, hi]
    lo, hi = group_boot(grp, lambda i: spearmanr(evo[i], y[i]).statistic)
    out["raw"]["evo_ci"] = [lo, hi]

    print("\n== 3. whole elements, between regions ==")
    print(f"  n = {out['n_elements']} distinct reference 200-mers")
    r = out["raw"]
    print(f"  raw Spearman  GC vs activity   : {r['gc_vs_activity']:+.4f}  CI [{r['gc_ci'][0]:+.3f}, {r['gc_ci'][1]:+.3f}]")
    print(f"  raw Spearman  Evo vs activity  : {r['evo_vs_activity']:+.4f}  CI [{r['evo_ci'][0]:+.3f}, {r['evo_ci'][1]:+.3f}]")
    print(f"  raw Spearman  Evo, active only : {r['evo_vs_activity_active_only']:+.4f}  (n={r['n_active']})")
    print(f"  raw Spearman  Evo vs GC        : {r['evo_vs_gc']:+.4f}")
    o = out["oof"]
    print(f"  OOF grouped 5-fold  GC        : rho {o['gc'][0]:+.4f}  RMSE {o['gc'][1]:.4f}")
    print(f"  OOF grouped 5-fold  Evo       : rho {o['evo'][0]:+.4f}  RMSE {o['evo'][1]:.4f}")
    print(f"  OOF grouped 5-fold  1/2/3-mer : rho {o['kmer_1_2_3'][0]:+.4f}  RMSE {o['kmer_1_2_3'][1]:.4f}")
    print(f"  OOF predict the training mean : RMSE {o['training_mean_rmse']:.4f}")
    return out


# ----------------------------------------------------------------- interaction

def reliability(eps, se):
    var = float(np.var(eps, ddof=1))
    noise = float(np.mean(se ** 2))
    rel = (var - noise) / var
    return var, noise, rel, float(np.sqrt(max(rel, 0.0)))


def interaction(df):
    eps = df["epsilon_recoded"].to_numpy(float)
    mod = df["model_recoded"].to_numpy(float)
    se = df["epsilon_se"].to_numpy(float)
    grp = df["group_id"].to_numpy()
    var, noise, rel, ceil = reliability(df["epsilon_refalt"].to_numpy(float), se)
    rho = spearmanr(mod, eps).statistic
    lo, hi = group_boot(grp, lambda i: spearmanr(mod[i], eps[i]).statistic)
    out = {"var_epsilon": var, "mean_se_sq": noise, "reliability": rel, "ceiling": ceil,
           "spearman": float(rho), "spearman_ci": [lo, hi],
           "pearson": float(pearsonr(mod, eps).statistic),
           "disattenuated_upper": float(hi / ceil)}
    print("\n== 4. interaction, against its ceiling ==")
    print(f"  Var(epsilon) {var:.5f}   mean SE^2 {noise:.5f}")
    print(f"  reliability {rel:.5f}   ceiling on any predictor's |r| {ceil:.5f}")
    print(f"  Evo Spearman {rho:+.5f}  95% CI [{lo:+.3f}, {hi:+.3f}]   Pearson {out['pearson']:+.5f}")
    print(f"  disattenuated upper bound on the true correlation {out['disattenuated_upper']:+.3f}")
    return out


# ------------------------------------------------------------------- detection

def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    order = pd.Series(score).rank().to_numpy()
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((order[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def detection(df):
    lab = (df["int_emVar"] == True).to_numpy()  # noqa: E712
    grp = df["group_id"].to_numpy()
    gc = gc_frac(df["seq_wt"])
    single = (df["y_a"].abs() + df["y_b"].abs()).to_numpy(float)
    closeness = -df["distance"].to_numpy(float)
    evo = np.abs(df["model_recoded"].to_numpy(float))

    singles_auroc = {"single_effect_size": auroc(single, lab), "evo_interaction_abs": auroc(evo, lab),
                     "closeness_neg_bp": auroc(closeness, lab), "gc_content": auroc(gc, lab)}

    def oof_auc(X):
        pred = np.empty(len(lab))
        for train, test in GroupKFold(n_splits=5).split(X, lab, grp):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[train], lab[train])
            pred[test] = m.predict_proba(X[test])[:, 1]
        return pred

    cov = np.column_stack([closeness, single, gc])
    p_cov = oof_auc(cov)
    p_all = oof_auc(np.column_stack([cov, evo]))
    a_cov, a_all = auroc(p_cov, lab), auroc(p_all, lab)
    lo, hi = group_boot(grp, lambda i: auroc(p_all[i], lab[i]) - auroc(p_cov[i], lab[i]))

    out = {"n_positives": int(lab.sum()), "single_feature_auroc": singles_auroc,
           "covariates_only": a_cov, "covariates_plus_evo": a_all,
           "gain": a_all - a_cov, "gain_ci": [lo, hi]}
    print("\n== 5. detection of the labelled non-additive pairs ==")
    print(f"  n = {len(df)}, positives = {out['n_positives']}")
    for k, v in singles_auroc.items():
        print(f"  AUROC  {k:<22s} {v:.4f}")
    print(f"  OOF logistic, distance + single-effect + GC : {a_cov:.5f}")
    print(f"  the same plus |Evo interaction|            : {a_all:.5f}")
    print(f"  gain from Evo {a_all - a_cov:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
    return out


# ------------------------------------------------------- precision stratification

def stratify(df, keeps=(1.0, 0.75, 0.5, 0.25)):
    rows = []
    print("\n== 6. nested subsets by measurement precision ==")
    print("  keep     n   reliability  ceiling   Evo rho   disattenuated")
    for k in keeps:
        cut = df["epsilon_se"].quantile(k)
        sub = df[df["epsilon_se"] <= cut]
        _, _, rel, ceil = reliability(sub["epsilon_refalt"].to_numpy(float),
                                      sub["epsilon_se"].to_numpy(float))
        rho = spearmanr(sub["model_recoded"], sub["epsilon_recoded"]).statistic
        rows.append({"keep": k, "n": int(len(sub)), "reliability": rel, "ceiling": ceil,
                     "spearman": float(rho), "disattenuated": float(rho / ceil)})
        print(f"  {k:>4.0%} {len(sub):>5d}      {rel:.3f}     {ceil:.3f}   {rho:+.4f}      {rho/ceil:+.4f}")
    return rows


# ------------------------------------------------------------------------ rmse

def rmse_block(df):
    eps = df["epsilon_recoded"].to_numpy(float)
    mod = df["model_recoded"].to_numpy(float)
    grp = df["group_id"].to_numpy()
    fold = df["fold"].to_numpy()
    cal = np.empty(len(eps))
    tmean = np.empty(len(eps))
    tmed = np.empty(len(eps))
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        m = LinearRegression().fit(mod[tr].reshape(-1, 1), eps[tr])
        cal[te] = m.predict(mod[te].reshape(-1, 1))
        tmean[te] = eps[tr].mean()
        tmed[te] = np.median(eps[tr])
    maj = np.sign(np.median(eps)) * np.ones(len(eps)) * 1e-9

    def rmse(p):
        return float(np.sqrt(np.mean((eps - p) ** 2)))

    def sign_acc(p):
        keep = np.abs(eps) >= 0.25
        return float((np.sign(p[keep]) == np.sign(eps[keep])).mean())

    def bal_sign(p):
        keep = np.abs(eps) >= 0.25
        pos, neg = keep & (eps > 0), keep & (eps < 0)
        return float(0.5 * ((np.sign(p[pos]) > 0).mean() + (np.sign(p[neg]) < 0).mean()))

    floor = float(np.sqrt(np.mean(df["epsilon_se"].to_numpy(float) ** 2)))
    zero = rmse(np.zeros(len(eps)))
    out = {"zero_baseline": zero, "perfect_floor": floor, "competitive_range": zero - floor,
           "calibrated_evo": rmse(cal), "training_mean": rmse(tmean), "training_median": rmse(tmed),
           "sign_accuracy": {"calibrated": sign_acc(cal), "training_mean": sign_acc(tmean),
                             "majority_sign": sign_acc(maj + np.median(eps))},
           "balanced_sign_accuracy": {"calibrated": bal_sign(cal), "training_mean": bal_sign(tmean)}}
    lo, hi = group_boot(grp, lambda i: float(np.sqrt(np.mean((eps[i] - tmean[i]) ** 2))
                                             - np.sqrt(np.mean((eps[i] - cal[i]) ** 2))))
    out["gain_over_training_mean_ci"] = [lo, hi]
    print("\n== 7. RMSE, with both ends of the scale ==")
    print(f"  predict zero                 {zero:.5f}   (mechanically the SD of the outcome)")
    print(f"  floor for a perfect predictor {floor:.5f}")
    print(f"  entire competitive range      {zero - floor:.5f}")
    print(f"  calibrated Evo               {out['calibrated_evo']:.5f}")
    print(f"  training-fold mean           {out['training_mean']:.5f}")
    print(f"  gain of Evo over that mean, 95% CI [{lo:+.6f}, {hi:+.6f}]")
    print(f"  raw sign accuracy: calibrated {out['sign_accuracy']['calibrated']:.4f}, "
          f"training mean {out['sign_accuracy']['training_mean']:.4f}")
    print(f"  balanced sign accuracy: calibrated {out['balanced_sign_accuracy']['calibrated']:.4f}, "
          f"training mean {out['balanced_sign_accuracy']['training_mean']:.4f}")
    return out


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    out_dir = args.out or (args.repo / "results/section4")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_flip(load(args.repo))
    numbers = {
        "reproduction": reproduction(df),
        "single_variants": singles(df),
        "elements": elements(df),
        "interaction": interaction(df),
        "detection": detection(df),
        "precision_stratification": stratify(df),
        "rmse": rmse_block(df),
        "libraries": df["pair_id"].str.split("|").str[0].str.split(";").str[-1]
                       .value_counts().to_dict(),
    }
    print("\n== libraries present in the retained set ==")
    print(" ", numbers["libraries"])
    path = out_dir / "numbers.json"
    path.write_text(json.dumps(numbers, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
