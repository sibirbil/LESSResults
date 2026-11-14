"""Synthetic heterogeneity experiment for LESS-B.

Data. J clusters in R^p with p >= 2J + 1. A random orthogonal matrix A splits
R^p into a centre block and a slope block:

    c_j     = a * A[:, j]                                  j = 1..J
    v_0..v_J = A[:, J], ..., A[:, 2J]
    theta_j = sqrt(kappa) v_0 + sqrt(1 - kappa) v_j

so ||theta_j|| = 1, <theta_i, theta_j> = kappa, and every theta_j is orthogonal
to every centre (theta_j . c_k = 0). Points of cluster j are drawn from
N(c_j, sigma_x^2 I) and

    y = <theta_j, x> + noise_std * eps.

kappa = 1 is one shared linear model, kappa = 0 gives orthogonal models.
a / sigma_x sets how far apart the clusters are.

Experiment. For every (kappa, sigma_x, replication) one training set and one
independent test set are drawn. On the test set the script records the MSE of

    OLS         global linear regression,
    BAYES       the true-parameter predictor sum_j P(j | x) <theta_j, x>,
    LESS-B*     the variants in VARIANTS, for every n_subsets in M_GRID, with
                linear local models and all other settings at the package
                defaults,

and merges one row per fit into logs/heterogeneity/results.csv, replacing the
rows of the methods it recomputes. One figure per LESS-B variant, in
plots/heterogeneity/, shows test MSE and the relative excess risk
(MSE - BAYES) / (OLS - BAYES) against n_subsets for every kappa, one panel per
sigma_x.

    PYTHONPATH=. python script_funcs/heterogeneity.py [--variants LESS-B LESS-B-noG]
                                                      [--reference] [--reps 20] [--n-jobs 8]

--cv scores every n_subsets by inner-fold MSE (the shuffled N_IN_CV-fold KFold
of main.py on the training set) into cv.csv; the n_subsets with the lowest
inner-fold MSE is the grid-search choice. selected_summary.csv and
plots/heterogeneity/selected_n_subsets.png give its test MSE / OLS and relative
excess risk over kappa and sigma_x (--summary redraws them).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from less import LESSBRegressor
from scipy.special import logsumexp
from scipy.stats import ortho_group
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import N_IN_CV, RANDOM_STATE

N_CLUSTERS = 5
N_FEATURES = 20
N_PER_CLUSTER = 400
N_TEST_PER_CLUSTER = 4000
A_SCALE = 1.0
NOISE_RATIO = 0.1  # noise_std = NOISE_RATIO * sigma_x
KAPPA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SIGMA_X_GRID = [1.0, 0.5, 0.3, 0.2]
M_GRID = [1, 2, 5, 10, 20, 50]
VARIANTS = {
    "LESS-B": {},
    "LESS-B-noG": {"global_estimator": None},
}

OUT_DIR = os.path.join("logs", "heterogeneity")
PLOT_DIR = os.path.join("plots", "heterogeneity")


def make_params(n_clusters, n_features, kappa, a, rng):
    """Return centres (p, J) and slopes (p, J) built from one orthogonal matrix."""
    J, p = n_clusters, n_features
    if p < 2 * J + 1:
        raise ValueError("n_features must be at least 2 * n_clusters + 1")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must lie in [0, 1]")
    A = ortho_group.rvs(p, random_state=rng)
    centres = a * A[:, :J]
    v0, V = A[:, [J]], A[:, J + 1 : 2 * J + 1]
    theta = np.sqrt(kappa) * v0 + np.sqrt(1.0 - kappa) * V
    return centres, theta


def sample(centres, theta, n_per_cluster, sigma_x, noise_std, rng):
    """Draw n_per_cluster points per cluster, rows interleaved over clusters."""
    p, J = centres.shape
    labels = np.tile(np.arange(J), n_per_cluster)
    X = centres.T[labels] + sigma_x * rng.standard_normal((labels.size, p))
    signal = np.einsum("ij,ji->i", X, theta[:, labels])
    y = signal + noise_std * rng.standard_normal(labels.size)
    return X, y, labels


def make_heterogeneous(
    n_clusters,
    n_features,
    n_per_cluster,
    kappa,
    a=1.0,
    sigma_x=1.0,
    noise_std=0.1,
    random_state=None,
):
    """Return X, y and cluster labels for one draw of the generator."""
    rng = np.random.default_rng(random_state)
    centres, theta = make_params(n_clusters, n_features, kappa, a, rng)
    return sample(centres, theta, n_per_cluster, sigma_x, noise_std, rng)


def bayes_predict(X, centres, theta, sigma_x):
    """Posterior-weighted true linear predictions under equal cluster priors."""
    sq = ((X[:, :, None] - centres[None]) ** 2).sum(axis=1)
    log_post = -sq / (2.0 * sigma_x**2)
    post = np.exp(log_post - logsumexp(log_post, axis=1, keepdims=True))
    return (post * (X @ theta)).sum(axis=1)


def scaled(estimator):
    """Standardise X and y around the estimator, as in main.py."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("regressor", estimator)]),
        transformer=StandardScaler(),
    )


def less(name, m):
    """LESS-B variant `name` with m subsets and linear local models."""
    return scaled(
        LESSBRegressor(
            n_subsets=m,
            local_estimator="linear",
            local_n_jobs=1,
            random_state=RANDOM_STATE,
            **VARIANTS[name],
        )
    )


def draw_cell(kappa, sigma_x, rep):
    """Parameters, training set and test set of one (kappa, sigma_x, rep) cell."""
    rng = np.random.default_rng(np.random.SeedSequence([RANDOM_STATE, rep]))
    centres, theta = make_params(N_CLUSTERS, N_FEATURES, kappa, A_SCALE, rng)
    noise_std = NOISE_RATIO * sigma_x
    X, y, _ = sample(centres, theta, N_PER_CLUSTER, sigma_x, noise_std, rng)
    X_te, y_te, _ = sample(centres, theta, N_TEST_PER_CLUSTER, sigma_x, noise_std, rng)
    return centres, theta, noise_std, X, y, X_te, y_te


def run_cell(kappa, sigma_x, rep, variants, reference):
    centres, theta, noise_std, X, y, X_te, y_te = draw_cell(kappa, sigma_x, rep)

    base = {"kappa": kappa, "sigma_x": sigma_x, "rep": rep, "noise_var": noise_std**2}
    rows = []

    def record(method, n_subsets, pred, sec):
        rows.append(
            {
                **base,
                "method": method,
                "n_subsets": n_subsets,
                "mse": float(np.mean((y_te - pred) ** 2)),
                "sec": sec,
            }
        )

    if reference:
        t = time.time()
        record("BAYES", np.nan, bayes_predict(X_te, centres, theta, sigma_x), time.time() - t)
        t = time.time()
        ols = scaled(LinearRegression()).fit(X, y)
        record("OLS", np.nan, ols.predict(X_te), time.time() - t)

    for name in variants:
        for m in M_GRID:
            t = time.time()
            model = less(name, m).fit(X, y)
            record(name, m, model.predict(X_te), time.time() - t)
    return rows


def cv_cell(kappa, sigma_x, rep, variants):
    """Inner-fold MSE of every n_subsets on the training set of one cell.

    The folds are the shuffled KFold of main.py, so the n_subsets with the
    lowest cv_mse is the grid-search choice of the main experiments.
    """
    _, _, _, X, y, _, _ = draw_cell(kappa, sigma_x, rep)
    folds = list(KFold(n_splits=N_IN_CV, shuffle=True, random_state=RANDOM_STATE).split(X))
    rows = []
    for name in variants:
        for m in M_GRID:
            err = [
                np.mean((y[va] - less(name, m).fit(X[tr], y[tr]).predict(X[va])) ** 2)
                for tr, va in folds
            ]
            rows.append(
                {
                    "kappa": kappa,
                    "sigma_x": sigma_x,
                    "rep": rep,
                    "method": name,
                    "n_subsets": m,
                    "cv_mse": float(np.mean(err)),
                }
            )
    return rows


def plot(df, method, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    sigmas = sorted(df.sigma_x.unique(), reverse=True)
    kappas = sorted(df.kappa.unique())
    x_ols, x_bayes = 0.6, M_GRID[-1] * 1.7
    fig, axes = plt.subplots(
        2, len(sigmas), figsize=(4.2 * len(sigmas), 8.0), constrained_layout=True, squeeze=False
    )
    mean = df.groupby(["sigma_x", "kappa", "method", "n_subsets"], dropna=False).mse.mean()
    for col, s in enumerate(sigmas):
        top, bottom = axes[0, col], axes[1, col]
        for colour, k in zip(ramp, kappas, strict=False):
            less = mean.loc[(s, k, method)]
            ols = mean.loc[(s, k, "OLS")].iloc[0]
            bayes = mean.loc[(s, k, "BAYES")].iloc[0]
            top.plot(
                less.index, less.values, color=colour, lw=2, marker="o", ms=5, label=f"κ = {k:g}"
            )
            top.scatter([x_ols], [ols], color=colour, marker="s", s=36, zorder=3)
            top.scatter([x_bayes], [bayes], color=colour, marker="_", s=200, lw=2, zorder=3)
            if k < 1.0:
                rel = (less.values - bayes) / (ols - bayes)
                bottom.plot(
                    less.index, rel, color=colour, lw=2, marker="o", ms=5, label=f"κ = {k:g}"
                )
        for ax in (top, bottom):
            ax.set_xscale("log")
            ax.grid(alpha=0.25, which="both")
            ax.set_xlabel("n_subsets (m)")
        top.set_yscale("log")
        top.set_xticks([x_ols, *M_GRID, x_bayes], ["OLS", *(str(m) for m in M_GRID), "Bayes"])
        top.set_title(f"{method}, σx = {s:g}  (a / σx = {A_SCALE / s:.3g})")
        bottom.set_xticks(M_GRID, [str(m) for m in M_GRID])
        bottom.axhline(1.0, color="#5f5e5a", lw=1, ls="--")
        bottom.axhline(0.0, color="#5f5e5a", lw=1, ls="--")
        bottom.set_ylim(-0.05, 1.15)
    axes[0, 0].set_ylabel("test MSE")
    axes[1, 0].set_ylabel(f"({method} − Bayes) / (OLS − Bayes)")
    axes[0, -1].legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=130)


def select(df, cv):
    """Test error at the n_subsets each variant picks by inner-fold MSE.

    One row per (method, kappa, sigma_x, rep) with the chosen n_subsets, its
    test MSE, MSE / OLS and the relative excess risk (MSE - BAYES) / (OLS - BAYES).
    """
    keys = ["kappa", "sigma_x", "rep"]
    best = cv.loc[cv.groupby(["method", *keys]).cv_mse.idxmin()]
    out = best.merge(df, on=["method", *keys, "n_subsets"])
    ref = df[df.method.isin(["OLS", "BAYES"])].pivot_table(index=keys, columns="method", values="mse")
    out = out.merge(ref.reset_index(), on=keys)
    out["ratio"] = out.mse / out.OLS
    out["rel"] = (out.mse - out.BAYES) / (out.OLS - out.BAYES)
    return out[["method", *keys, "n_subsets", "mse", "OLS", "BAYES", "ratio", "rel"]]


def summarise(sel):
    """Mean and 95% half-width over replications per (method, kappa, sigma_x)
    of MSE / OLS (ratio) and of the relative excess risk (rel), with the median
    chosen n_subsets. rel is undefined at kappa = 1, where OLS is the Bayes rule.
    """
    g = sel.groupby(["method", "kappa", "sigma_x"])
    out = g.n_subsets.median().rename("median_m").to_frame()
    for col in ("ratio", "rel"):
        out[col] = g[col].mean()
        out[f"{col}_ci"] = 1.96 * g[col].std() / np.sqrt(g[col].count())
    out.loc[out.index.get_level_values("kappa") == 1.0, ["rel", "rel_ci"]] = np.nan
    return out.reset_index()


def plot_selected(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    methods = sorted(table.method.unique())
    kappas = sorted(table.kappa.unique())
    fig, axes = plt.subplots(
        1, len(methods), figsize=(4.6 * len(methods), 4.0), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods, strict=True):
        for colour, k in zip(ramp, kappas, strict=False):
            t = table[(table.method == method) & (table.kappa == k)].sort_values("sigma_x")
            ax.errorbar(
                A_SCALE / t.sigma_x.to_numpy(),
                t.ratio,
                yerr=t.ratio_ci,
                color=colour,
                lw=2,
                marker="o",
                ms=5,
                capsize=3,
                label=f"κ = {k:g}",
            )
        ax.axhline(1.0, color="#5f5e5a", lw=1, ls="--")
        ax.set_xticks(A_SCALE / np.array(SIGMA_X_GRID))
        ax.set_xticklabels([f"{A_SCALE / s:.3g}" for s in SIGMA_X_GRID])
        ax.set_xlabel("a / σx  (cluster separation)")
        ax.set_title(f"{method}, n_subsets chosen by {N_IN_CV}-fold CV")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 1.2)
    axes[0].set_ylabel("test MSE / OLS test MSE")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.savefig(path, dpi=130)


def write_summary():
    df = pd.read_csv(os.path.join(OUT_DIR, "results.csv"))
    cv = pd.read_csv(os.path.join(OUT_DIR, "cv.csv"))
    table = summarise(select(df, cv))
    table.to_csv(os.path.join(OUT_DIR, "selected_summary.csv"), index=False)
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_selected(table, os.path.join(PLOT_DIR, "selected_n_subsets.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--reference", action="store_true", help="recompute OLS and BAYES")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--cv", action="store_true", help="inner-fold MSE of every n_subsets -> cv.csv"
    )
    parser.add_argument(
        "--summary", action="store_true", help="only redraw the CV-selected table and figure"
    )
    args = parser.parse_args()
    if args.summary:
        write_summary()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    cells = [(k, s, r) for k in KAPPA_GRID for s in SIGMA_X_GRID for r in range(args.reps)]
    if args.cv:
        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(cv_cell)(*c, args.variants) for c in cells
        )
        pd.DataFrame([row for rows in out for row in rows]).to_csv(
            os.path.join(OUT_DIR, "cv.csv"), index=False
        )
        write_summary()
        return

    path = os.path.join(OUT_DIR, "results.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    reference = args.reference or old.empty or not {"OLS", "BAYES"} <= set(old.method)

    out = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_cell)(*c, args.variants, reference) for c in cells
    )
    new = pd.DataFrame([row for rows in out for row in rows])
    if not old.empty:
        old = old[~old.method.isin(new.method.unique())]
    df = pd.concat([old, new], ignore_index=True)
    df.to_csv(path, index=False)
    for name in sorted(set(df.method) & set(VARIANTS)):
        plot(df, name, os.path.join(PLOT_DIR, f"mse_vs_n_subsets_{name}.png"))


if __name__ == "__main__":
    main()
