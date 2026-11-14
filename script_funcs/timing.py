"""Measures fit and predict times of each model on a few problems.

Uses a monotonic clock, repeats every fit and keeps the median, and refuses to
start when the machine is busy. Results go to logs/timing/.

    PYTHONPATH=. python script_funcs/timing.py [--datasets energy] [--repeats 3]
"""

import argparse
import json
import os
import statistics
import time

import psutil
from less import LESSARegressor, LESSBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.datasets import casp, ctslices, energy, housing
from src.models.locreg import LocalRegression
from src.models.magging import Magging
from src.models.tsk import TSKRegressorPipeline

# --- Configuration ---
ALL_DATASETS = {
    "energy": energy,
    "housing": housing,
    # ctslices is 53 500 x 384, the widest problem in the suite; casp is
    # 45 730 x 9.
    "ctslices": ctslices,
    "casp": casp,
}
# Directory holding the dataset CSV files.
DATA_DIR = "datasetsR"
LOG_DIR = "logs/timing"

LESS_MODELS = {
    "LESSARegressor": LESSARegressor,
    "LESSBRegressor": LESSBRegressor,
}

OTHER_MODELS = {
    "TSK": TSKRegressorPipeline(consequent=RandomForestRegressor()),
    "RF": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(),
    # max_iter=5000 so the optimiser converges.
    "MLP": MLPRegressor(max_iter=5000),
    "LOCR": LocalRegression(),
    "MAG": Magging(),
    "GPR": GaussianProcessRegressor(),
    "XGB": XGBRegressor(),
    "LGBM": LGBMRegressor(),
}

N_SUBSETS_RANGE = range(0, 101, 5)
# A method without four folds on a problem in the main run is not timed there;
# these are the reasons recorded for it.
STRUCTURAL = {
    "GPR": "kernel matrix is n^2 doubles: {gb:.0f} GB at n = {n}",
    "LOCR": "cost is order n^2 d^2, at n = {n}, d = {d}",
    "TSK": "fuzzy c-means per grid candidate at n = {n}",
    "ANFIS": "rule count is 2^d, d = {d}",
}
# Timing does not start when CPU utilisation (percent) is above this.
BUSY_LIMIT = 50


# The results file a dataset's best parameters come from. Only the name differs.
RESULT_NAME = {"energy": "ENERGY", "housing": "HOUSING",
               "ctslices": "CTSLICES_GROUPED", "casp": "CASP"}
RESULT_DIR = "logs/results"


def _best_params(dataset_name, model_name):
    """The configuration the nested CV selected for this model and dataset.

    The most frequent setting across the outer folds is used, with ties going
    to the earliest fold. Returns None when no selection is recorded.
    """
    path = os.path.join(RESULT_DIR, f"{RESULT_NAME[dataset_name]}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        metrics = json.load(fh).get("_metrics", {}).get(model_name, {})
    seen = {}
    for fold in sorted(metrics, key=lambda k: int(k)):
        bp = metrics[fold].get("best_params")
        if not bp:
            continue
        key = tuple(sorted((k, str(v)) for k, v in bp.items()))
        seen.setdefault(key, [0, bp])
        seen[key][0] += 1
    if not seen:
        return None
    return max(seen.values(), key=lambda c: c[0])[1]


def _ran_in_comparison(dataset_name, model_name):
    """Did this method produce four folds on this problem in the main run?
    True when the results file is absent."""
    path = os.path.join(RESULT_DIR, f"{RESULT_NAME[dataset_name]}.json")
    if not os.path.exists(path):
        return True
    with open(path) as fh:
        folds = json.load(fh).get(model_name)
    return isinstance(folds, dict) and len(folds) == 4


def _save(path, results):
    """Write after every model, so a killed run resumes instead of restarting."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=4)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _timed(fit, repeats):
    """Median of `repeats` fits, with every reading kept."""
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        fit()
        runs.append(time.perf_counter() - start)
    return statistics.median(runs), runs


# --- Main Execution ---
def run_timing_experiment(datasets, repeats):
    """
    Measures the fitting time for LESS models and other regressors
    on different datasets.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    # Refuse to start when CPU utilisation is above BUSY_LIMIT.
    busy = psutil.cpu_percent(interval=2.0)
    if busy > BUSY_LIMIT:
        raise SystemExit(
            f"cpu is {busy:.0f}% busy, above {BUSY_LIMIT}%: something else is "
            "running and every number here would be a measurement of it"
        )
    print(f"cpu {busy:.0f}% busy, {repeats} repeats per fit")

    for dataset_name, load_func in datasets.items():
        print(f"--- Processing Dataset: {dataset_name.upper()} ---")

        try:
            X, y = load_func(DATA_DIR)
        except FileNotFoundError:
            print(
                f"  ERROR: Could not find data for {dataset_name} in "
                f"'{DATA_DIR}' directory. Skipping."
            )
            continue

        output_path = os.path.join(LOG_DIR, f"{dataset_name.upper()}.json")
        if os.path.exists(output_path):
            with open(output_path) as fh:
                dataset_results = json.load(fh)
            print(f"  resuming: {len([k for k in dataset_results if not k.startswith('_')])} "
                  f"entries already recorded")
        else:
            dataset_results = {}

        # --- Time LESS Models ---
        for model_name, model_class in LESS_MODELS.items():
            print(f"  --- Testing Model: {model_name} ---")
            model_timings = dataset_results.get(model_name, {})
            for requested in N_SUBSETS_RANGE:
                # the sweep starts at 0, which LESS rejects; one subset is the
                # smallest meaningful setting
                n_subsets = max(requested, 1)

                if f"n_subsets_{n_subsets}" in model_timings:
                    continue
                print(f"    n_subsets = {n_subsets}")

                box = {}

                def fit_once(cls=model_class, m=n_subsets, box=box):
                    model = TransformedTargetRegressor(
                        regressor=Pipeline([("scaler", StandardScaler()),
                                            ("regressor", cls(n_subsets=m))]),
                        transformer=StandardScaler(),
                    )
                    model.fit(X, y)
                    box["model"] = model

                duration, runs = _timed(fit_once, repeats)
                spread = (max(runs) - min(runs)) / duration if duration else 0
                print(f"      Fit time: {duration:.4f} s (spread {spread:.1%})")
                model_timings[f"n_subsets_{n_subsets}"] = duration
                dataset_results.setdefault("_runs", {})[
                    f"{model_name}_n_subsets_{n_subsets}"] = runs
                # Predict time is recorded as well.
                start = time.perf_counter()
                box["model"].predict(X)
                dataset_results.setdefault("_less_predict", {}).setdefault(
                    model_name, {})[f"n_subsets_{n_subsets}"] = \
                    time.perf_counter() - start
                dataset_results[model_name] = model_timings
                _save(output_path, dataset_results)

            dataset_results[model_name] = model_timings
            _save(output_path, dataset_results)

        # --- Time Other Models ---
        for model_name, estimator in OTHER_MODELS.items():
            if model_name in dataset_results or \
                    model_name in dataset_results.get("_refused", {}):
                continue
            print(f"  --- Testing Model: {model_name} ---")

            if not _ran_in_comparison(dataset_name, model_name):
                reason = STRUCTURAL.get(
                    model_name, "no result in the main comparison"
                ).format(n=len(X), d=X.shape[1], gb=len(X) ** 2 * 8 / 2**30)
                print(f"      refused: {reason}")
                dataset_results.setdefault("_refused", {})[model_name] = reason
                _save(output_path, dataset_results)
                continue

            chosen = _best_params(dataset_name, model_name)
            if chosen:
                # An estimator-valued hyperparameter is stored as its repr, and a
                # repr is not an estimator. TSK's consequent is the only one; it
                # is dropped and the constructed consequent is timed.
                dropped = {k: v for k, v in chosen.items()
                           if isinstance(v, str) and v.endswith(")")}
                chosen = {k: v for k, v in chosen.items() if k not in dropped}
                for k, v in dropped.items():
                    print(f"      keeping constructed {k} ({v})")
                try:
                    estimator = clone(estimator).set_params(**chosen)
                    print(f"      best params: {chosen}")
                    dataset_results.setdefault("_params", {})[model_name] = chosen
                except ValueError as exc:
                    print(f"      keeping defaults, {exc}")
            else:
                print("      no recorded selection, keeping package defaults")

            fitted = {}

            def fit_once(est=estimator, box=fitted):
                model = TransformedTargetRegressor(
                    regressor=Pipeline([("scaler", StandardScaler()),
                                        ("regressor", clone(est))]),
                    transformer=StandardScaler(),
                )
                model.fit(X, y)
                box["model"] = model

            duration, runs = _timed(fit_once, repeats)
            spread = (max(runs) - min(runs)) / duration if duration else 0
            print(f"      Fit time: {duration:.4f} s (spread {spread:.1%})")

            # Methods such as LocR and KNN do most of their work at predict
            # time, so predict time is recorded alongside fit time.
            start = time.perf_counter()
            fitted["model"].predict(X)
            predict_time = time.perf_counter() - start
            print(f"      Predict time: {predict_time:.4f} s")
            dataset_results.setdefault("_predict", {})[model_name] = predict_time

            dataset_results[model_name] = duration
            dataset_results.setdefault("_runs", {})[model_name] = runs
            _save(output_path, dataset_results)

        _save(output_path, dataset_results)
        print(f"\nResults for {dataset_name.upper()} saved to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["energy"],
                    choices=list(ALL_DATASETS),
                    help="default: energy, the problem the figure reports")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    print("Starting timing experiment...")
    run_timing_experiment({k: ALL_DATASETS[k] for k in args.datasets}, args.repeats)
    print("\nTiming experiment finished.")
