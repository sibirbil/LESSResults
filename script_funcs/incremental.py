"""Incremental study for LESS-A and LESS-B.

Scores every combination of a linear or tree local estimator with a linear or
random-forest global estimator. Results go to logs/incremental/.
"""

import argparse
import json
import logging
import os
import warnings
from collections import defaultdict

import numpy as np
from less import LESSARegressor, LESSBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    N_IN_CV,
    N_OUT_CV,
    RANDOM_STATE,
    SCORING,
)
from src.problems import ALL_PROBLEMS, ROW_IDENTITY, row_identity_groups
from src.protocol import check as protocol_check

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

LOG_DIR = "logs"
GRID = {
    "n_subsets": [10, 20],
    "kernel_coeff": [0.1, 0.01],
}
os.makedirs(LOG_DIR, exist_ok=True)
_ap = argparse.ArgumentParser()
_ap.add_argument("--problems", nargs="+", default=list(ALL_PROBLEMS),
                 choices=list(ALL_PROBLEMS),
                 help="default: every problem the tables score")
_ap.add_argument("--results-dir", default="incremental",
                 help="subdirectory of logs/ to write to")
_ap.add_argument("--n-jobs", type=int, default=4,
                 help="GridSearchCV workers per problem")
_ap.add_argument("--log-file", default="training.log",
                 help="log file under logs/; a rerun should not append "
                      "to the log of the run it replaces")
_args, _ = _ap.parse_known_args()
RESULT_DIR = os.path.join(LOG_DIR, _args.results_dir)
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, _args.log_file)),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    protocol_check(RESULT_DIR,
               {"inner_cv": "same policy as the outer split"})


    for problem_name in _args.problems:
        problem, group_spec = ALL_PROBLEMS[problem_name]
        logger.info(f"Starting problem: {problem_name}")

        result_path = os.path.join(RESULT_DIR, f"{problem_name}.json")

        grid_models = [
            (
                LESSARegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="linear",
                    global_estimator=LinearRegression,
                ),
                "AV-D",
            ),
            (
                LESSARegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="tree",
                    global_estimator=LinearRegression,
                ),
                "AV-TL",
            ),
            (
                LESSARegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="linear",
                ),
                "AV-LXGB",
            ),
            (
                LESSARegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="tree",
                ),
                "AV-TXGB",
            ),
            (
                LESSBRegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="linear",
                    global_estimator=LinearRegression,
                ),
                "B-D",
            ),
            (
                LESSBRegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="tree",
                    global_estimator=LinearRegression,
                ),
                "B-TL",
            ),
            (
                LESSBRegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="linear",
                ),
                "B-LXGB",
            ),
            (
                LESSBRegressor(
                    random_state=RANDOM_STATE,
                    local_estimator="tree",
                ),
                "B-TXGB",
            ),
        ]

        # Load existing results
        if os.path.exists(result_path):
            with open(result_path) as f:
                previous_results = json.load(f)
        else:
            previous_results = {}

        X, y = problem(DATA_PATH)
        if group_spec is None:
            groups = None
        elif group_spec is ROW_IDENTITY:
            groups = row_identity_groups(X)
        else:
            groups = group_spec(DATA_PATH)

        cachedir = os.path.join(DATA_PATH, "cache")

        scores = defaultdict(list)

        # The inner split follows the outer one: seeded shuffled KFold, or
        # GroupKFold where the data defines groups.
        # GroupKFold is left unshuffled: sklearn assigns groups to folds
        # deterministically and takes no random_state. Same policy as main.py.
        outer_fold = (
            GroupKFold(n_splits=N_OUT_CV) if groups is not None
            else KFold(n_splits=N_OUT_CV, shuffle=True, random_state=RANDOM_STATE)
        )

        for fold_num, (train_index, test_index) in enumerate(
            outer_fold.split(X, y, groups=groups), start=1
        ):
            logger.info(f"Fold {fold_num} for {problem_name}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            g_train = groups[train_index] if groups is not None else None

            for estimator, estimator_type in grid_models:
                if str(fold_num) in previous_results.get(estimator_type, {}):
                    logger.info(
                        f"Skipping {estimator_type} fold {fold_num} (already computed)."
                    )
                    continue

                logger.info(f"Training {estimator_type} on fold {fold_num}...")

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("regressor", estimator),
                    ],
                )

                model = TransformedTargetRegressor(
                    regressor=pipe,
                    transformer=StandardScaler(),
                )

                gcv = GridSearchCV(
                    estimator=model,
                    param_grid={
                        f"regressor__regressor__{key}": val for key, val in GRID.items()
                    },
                    # Each GridSearchCV worker holds its own copy of the fold.
                    n_jobs=_args.n_jobs,
                    cv=(GroupKFold(N_IN_CV) if g_train is not None
                        else KFold(n_splits=N_IN_CV, shuffle=True,
                                   random_state=RANDOM_STATE)),
                    scoring=SCORING,
                    verbose=1,
                    refit=True,
                )

                # A cell that raises or predicts non-finite values is recorded
                # under "_failed" with its reason, and the loop continues.
                try:
                    gcv.fit(X_train, y_train, groups=g_train)
                    gcv_pred = gcv.best_estimator_.predict(X_test)
                    if not np.all(np.isfinite(gcv_pred)):
                        bad = int(np.sum(~np.isfinite(gcv_pred)))
                        raise ValueError(
                            f"{bad} of {len(gcv_pred)} predictions are not finite"
                        )
                    mse = mean_squared_error(gcv_pred, y_test)
                except Exception as exc:
                    logger.warning(
                        f"{estimator_type} failed on fold {fold_num}: {exc}"
                    )
                    previous_results.setdefault("_failed", {}).setdefault(
                        estimator_type, {})[str(fold_num)] = {
                            "reason": f"{type(exc).__name__}: {exc}"[:400]}
                    _tmp = f"{result_path}.tmp"
                    with open(_tmp, "w") as fh:
                        json.dump(previous_results, fh, indent=2)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(_tmp, result_path)
                    continue
                logger.info(f"{estimator_type} MSE on fold {fold_num}: {mse:.4f}")

                # Save the result
                if estimator_type not in previous_results:
                    previous_results[estimator_type] = {}
                previous_results[estimator_type][str(fold_num)] = mse

                # Record the selected hyperparameters for each cell.
                previous_results.setdefault("_best_params", {}).setdefault(
                    estimator_type, {})[str(fold_num)] = {
                        k.replace("regressor__regressor__", ""): v
                        for k, v in gcv.best_params_.items()
                }

                # Write to a temporary file and rename it, so an interrupted run
                # does not leave a partial results file.
                _tmp = f"{result_path}.tmp"
                with open(_tmp, "w") as f:
                    json.dump(previous_results, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(_tmp, result_path)
