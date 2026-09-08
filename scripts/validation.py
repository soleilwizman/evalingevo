"""Grouped evaluation shared by all benchmark readouts.

Both hyperparameter selection and preprocessing are fitted inside genomic-group
splits. Historical results using RidgeCV's ungrouped inner loop are not the same
protocol, even when their outer folds happen to coincide.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 6, 25)
PROTOCOL = "nested-genomic-group-cv-v2"


def group_splits(groups, folds=5, seed=0):
    groups = np.asarray(groups)
    if groups.ndim != 1 or pd.isna(groups).any():
        raise ValueError("groups must be a complete one-dimensional array")
    if not isinstance(folds, (int, np.integer)) or not 2 <= folds <= len(np.unique(groups)):
        raise ValueError("folds must be between 2 and the number of genomic groups")
    if seed is None:
        raise ValueError("an explicit fold seed is required")
    splitter = GroupKFold(n_splits=int(folds), shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((len(groups), 1)), groups=groups))


def checked_arrays(X, y, groups):
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or not X.shape[1] or y.ndim != 1 or len(X) != len(y) or len(y) != len(groups):
        raise ValueError("features, targets and groups must have aligned, nonempty rows")
    if not len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be nonempty and finite")
    return X, y, groups


def fit_ridge(X, y, groups, folds=5, seed=0, alphas=ALPHAS):
    X, y, groups = checked_arrays(X, y, groups)
    inner_folds = min(folds, len(np.unique(groups)))
    splits = group_splits(groups, inner_folds, seed)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": np.asarray(alphas, float)},
        scoring="neg_mean_squared_error",
        cv=splits,
        error_score="raise",
        n_jobs=1,
    )
    return search.fit(X, y)


def out_of_fold(X, y, groups, folds=5, seed=0):
    X, y, groups = checked_arrays(X, y, groups)
    prediction = np.empty(len(y))
    for train, test in group_splits(groups, folds, seed):
        model = fit_ridge(X[train], y[train], groups[train], folds, seed)
        prediction[test] = model.predict(X[test])
    return prediction


def out_of_fold_linear(features, y, groups, folds=5, ridge=False, seed=0):
    if ridge:
        return out_of_fold(features, y, groups, folds, seed)
    return out_of_fold_estimator(
        features,
        y,
        groups,
        make_pipeline(StandardScaler(), LinearRegression()),
        folds,
        seed,
    )


def out_of_fold_estimator(X, y, groups, estimator, folds=5, seed=0, probability=False):
    X, y, groups = checked_arrays(X, y, groups)
    prediction = np.empty(len(y))
    for train, test in group_splits(groups, folds, seed):
        model = clone(estimator).fit(X[train], y[train])
        if probability:
            if not np.array_equal(model.classes_, [0, 1]):
                raise ValueError("each training fold must contain both binary classes")
            prediction[test] = model.predict_proba(X[test])[:, 1]
        else:
            prediction[test] = model.predict(X[test])
    return prediction


def mean_prediction(y, groups, folds=5, seed=0):
    y = np.asarray(y, float)
    prediction = np.empty(len(y))
    for train, test in group_splits(groups, folds, seed):
        prediction[test] = y[train].mean()
    return prediction


def selected_predictions(features, y, groups, folds=5, seed=0):
    """Choose the representation and alpha using only each outer training set."""
    prediction = np.empty(len(y))
    selected = []
    arrays = {name: checked_arrays(X, y, groups)[0] for name, X in features.items()}
    y, groups = np.asarray(y), np.asarray(groups)
    if not arrays:
        raise ValueError("at least one candidate representation is required")
    for train, test in group_splits(groups, folds, seed):
        candidates = {
            name: fit_ridge(X[train], y[train], groups[train], folds, seed)
            for name, X in arrays.items()
        }
        name = max(candidates, key=lambda key: candidates[key].best_score_)
        prediction[test] = candidates[name].predict(arrays[name][test])
        selected.append(name)
    return prediction, selected


def fit_residual(X, baseline, y, groups, folds=5, seed=0, alphas=ALPHAS):
    """Tune residual ridge with the nuisance baseline refitted inside each split.

    Cross-fitting once before GridSearchCV would let its validation labels affect
    training residuals. Rebuild those residuals for each inner training subset.
    """
    errors = np.zeros(len(alphas))
    for train, test in group_splits(groups, min(folds, len(np.unique(groups))), seed):
        train_folds = min(folds, len(np.unique(groups[train])))
        crossfit = out_of_fold(baseline[train], y[train], groups[train], train_folds, seed)
        nuisance = fit_ridge(baseline[train], y[train], groups[train], folds, seed)
        train_target = y[train] - crossfit
        test_target = y[test] - nuisance.predict(baseline[test])
        for index, alpha in enumerate(alphas):
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                X[train], train_target
            )
            errors[index] += np.sum((candidate.predict(X[test]) - test_target) ** 2)
    crossfit = out_of_fold(baseline, y, groups, min(folds, len(np.unique(groups))), seed)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alphas[np.argmin(errors)]))
    return model.fit(X, y - crossfit)


def residual_predictions(X, baseline, y, groups, folds=5, seed=0):
    """Cross-fit both stages, with no train-target leakage at either CV level."""
    X, y, groups = checked_arrays(X, y, groups)
    baseline, _, _ = checked_arrays(baseline, y, groups)
    prediction, target = np.empty(len(y)), np.empty(len(y))
    for train, test in group_splits(groups, folds, seed):
        baseline_model = fit_ridge(baseline[train], y[train], groups[train], folds, seed)
        residual_model = fit_residual(
            X[train], baseline[train], y[train], groups[train], folds, seed
        )
        prediction[test] = residual_model.predict(X[test])
        target[test] = y[test] - baseline_model.predict(baseline[test])
    return prediction, target


def permutation_null(X, y, groups, folds=5, seed=0, draws=20):
    """Block permutation null, exchanging groups only with groups of equal size.

    This assumes exchangeable groups within each size stratum; it preserves
    within-group label dependence and evaluates against the permuted target.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    rng = np.random.default_rng(seed)
    strata = {}
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        strata.setdefault(len(rows), []).append(rows)
    if draws and not any(len(blocks) > 1 for blocks in strata.values()):
        raise ValueError("no equal-sized genomic groups can be exchanged for the null")
    values = []
    for _ in range(draws):
        shuffled = y.copy()
        for blocks in strata.values():
            for destination, source in zip(blocks, rng.permutation(len(blocks))):
                shuffled[destination] = y[blocks[source]]
        values.append(
            float(spearmanr(out_of_fold(X, shuffled, groups, folds, seed), shuffled).statistic)
        )
    return values
