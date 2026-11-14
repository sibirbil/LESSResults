"""Shared settings: data path, random seed, cross-validation sizes, scoring and
the hyperparameter grid of every model.
"""

# PATH
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# Resolved against this file, so the repository works wherever it is cloned.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasetsR")

# SEED
RANDOM_STATE = 16252329
N_OUT_CV = 4
N_IN_CV = 4

# The inner cross-validation selects hyperparameters by the same quantity the
# tables report. GridSearchCV's default, the estimator's own score(), is R^2 =
# 1 - MSE / mean((y - ybar)^2); averaged over folds, that weights each fold's
# MSE by the inverse of its target variance.
SCORING = "neg_mean_squared_error"

# LESS-B: number of subsets, kernel width and local learner.
LESSB_GRID = {
    "n_subsets": [10, 20],
    "kernel_coeff": [0.1, 0.01],
    "local_estimator": ["linear", "tree"],
    "random_state": [RANDOM_STATE],
}
# LESS-A searches the same three axes as LESS-B.
LESSA_GRID = {
    "n_subsets": [10, 20],
    "kernel_coeff": [0.1, 0.01],
    "local_estimator": ["linear", "tree"],
    "random_state": [RANDOM_STATE],
}

# The forest uses the same tree counts and depths as the boosting baselines.
RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8],
    "random_state": [RANDOM_STATE],
}

KNN_GRID = {
    "n_neighbors": [3, 5, 10, 20, 50, 100],
}
# max_iter is an upper bound; adam stops on its own (n_iter_no_change=10,
# tol=1e-4).
MLP_GRID = {"alpha": [0.0001, 0.001, 0.01], "max_iter": [5000]}
LOCR_GRID = {"frac": [0.01, 0.05, 0.10, 0.20]}
MAG_GRID = {"frac": [0.10, 0.20, 0.30], "random_state": [RANDOM_STATE]}

GPR_GRID = {"alpha": [1.0e-1, 1.0e-4, 1.0e-8], "random_state": [RANDOM_STATE]}
LGBM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    # lightgbm grows leaf-wise, so num_leaves alone does not bound depth.
    # max_depth follows xgboost's grid and num_leaves is an upper bound.
    "max_depth": [6, 8],
    "num_leaves": [255],
    "random_state": [RANDOM_STATE],
    "verbose": [-1],
}

# xgboost grows depthwise and max_leaves is unset, so max_depth alone bounds
# the tree.
XGB_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [6, 8],
    "random_state": [RANDOM_STATE],
    "verbosity": [0],
}

# The tree consequents and the fuzzy clustering are seeded so that a recorded
# score is reproducible.
TSK_GRID = {
    "consequent": [Ridge(),
                   DecisionTreeRegressor(random_state=RANDOM_STATE),
                   RandomForestRegressor(random_state=RANDOM_STATE)],
    "n_cluster": [5, 10, 20],
    "fuzzy_index": [2.0, "auto"],
    "random_state": [RANDOM_STATE],
}

ANFIS_GRID = {"hybrid": [True, False], "epoch": [25, 50]}
