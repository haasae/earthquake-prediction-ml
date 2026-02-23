from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class TrainResult:
    model: Any
    best_params: Dict[str, Any] | None
    cv_rmse_mean: float | None
    cv_rmse_std: float | None


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    base_params: Dict[str, Any],
    tuning_enabled: bool,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    random_state: int,
) -> TrainResult:
    rf = RandomForestRegressor(random_state=random_state, **base_params)

    if not tuning_enabled:
        rf.fit(X_train, y_train)
        return TrainResult(model=rf, best_params=None, cv_rmse_mean=None, cv_rmse_std=None)

    # Conservative search space so Windows laptops don't look "frozen"
    param_dist = {
        "n_estimators": [150, 300, 600],
        "max_depth": [None, 6, 10, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        n_jobs=n_jobs,
        scoring="neg_root_mean_squared_error",
        verbose=2,
        random_state=random_state,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    best.fit(X_train, y_train)

    # CV score is negative RMSE
    cv_rmse = -search.best_score_

    # Estimate std from cv results if present
    std = None
    if "std_test_score" in search.cv_results_:
        # std_test_score corresponds to neg RMSE; take the best candidate's std
        best_idx = int(search.best_index_)
        std = float(search.cv_results_["std_test_score"][best_idx])
        std = abs(std)

    return TrainResult(
        model=best,
        best_params=search.best_params_,
        cv_rmse_mean=float(cv_rmse),
        cv_rmse_std=std,
    )
