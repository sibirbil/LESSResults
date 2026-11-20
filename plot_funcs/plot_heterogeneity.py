"""Plots the synthetic heterogeneity experiment (logs/heterogeneity) to
plots/heterogeneity/.

    mse_vs_n_subsets_<variant>.png   test MSE and the relative excess risk
                                     (MSE - Bayes) / (OLS - Bayes) against the
                                     number of subsets, one column per sigma_x
    selected_n_subsets.png           test MSE / OLS test MSE at the number of
                                     subsets chosen by cross-validation
    example_data.png                 the inputs and y against <theta_1, x> for
                                     two cluster separations and three kappa
    example_fit.png                  true against predicted y on a test set

The two example figures use a small instance (two clusters, five features)
of the generator in script_funcs/heterogeneity.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

from script_funcs import heterogeneity as h

sns.set_theme(style="whitegrid", color_codes=True)
sns.set_style("ticks")

# Every figure is FIG_WIDTH inches wide (selected_n_subsets.png 0.85 of it,
# as it is printed at 0.85 of the text width), so the fonts print at the same
# size in all of them.
SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 30
FIG_WIDTH = 19.0
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=MEDIUM_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

SAVE_DIR = h.PLOT_DIR
# One shade per kappa, light (kappa = 0) to dark (kappa = 1).
KAPPA_COLOURS = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
CLUSTER_COLOURS = ["#1f77b4", "#ff7f0e"]
REFERENCE_COLOUR = "#5f5e5a"
VARIANT_LABELS = {"LESS-B": "LESS-B", "LESS-B-noG": "LESS-B (W-NoG)"}


def _grid(ax):
    ax.grid(True, "major", ls="--", lw=0.5, c="k", alpha=0.3)


def _save(fig, name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _kappa_handles(kappas):
    return [
        Line2D([], [], color=c, lw=2.5, marker="o", ms=7, label=rf"$\kappa = {k:g}$")
        for c, k in zip(KAPPA_COLOURS, kappas, strict=False)
    ]


def plot_mse_vs_n_subsets(df, method):
    """Test MSE (top) and relative excess risk (bottom) against n_subsets."""
    sigmas = sorted(df.sigma_x.unique(), reverse=True)
    kappas = sorted(df.kappa.unique())
    x_ols, x_bayes = 0.55, h.M_GRID[-1] * 1.8
    mean = df.groupby(["sigma_x", "kappa", "method", "n_subsets"], dropna=False).mse.mean()

    fig, axes = plt.subplots(2, len(sigmas), figsize=(FIG_WIDTH, 10.5), squeeze=False)
    rel_max = 1.0
    for col, s in enumerate(sigmas):
        top, bottom = axes[0, col], axes[1, col]
        for colour, k in zip(KAPPA_COLOURS, kappas, strict=False):
            less = mean.loc[(s, k, method)]
            ols = mean.loc[(s, k, "OLS")].iloc[0]
            bayes = mean.loc[(s, k, "BAYES")].iloc[0]
            top.plot(less.index, less.values, color=colour, lw=2.5, marker="o", ms=7)
            top.scatter([x_ols], [ols], color=colour, marker="s", s=60, zorder=3)
            top.scatter([x_bayes], [bayes], color=colour, marker="_", s=300, lw=3, zorder=3)
            if k < 1.0:
                # Skipped at kappa = 1, where OLS - Bayes is close to zero.
                rel = (less.values - bayes) / (ols - bayes)
                bottom.plot(less.index, rel, color=colour, lw=2.5, marker="o", ms=7)
                rel_max = max(rel_max, rel.max())
        for ax in (top, bottom):
            ax.set_xscale("log")
            ax.minorticks_off()
            _grid(ax)
        bottom.set_xlabel(r"Number of subsets ($m$)")
        top.set_yscale("log")
        # OLS and Bayes are drawn left and right of the curves and named in
        # the legend.
        top.set_xticks(h.M_GRID, [str(m) for m in h.M_GRID])
        top.set_title(rf"$a/\sigma_x = {h.A_SCALE / s:.3g}$")
        bottom.set_xticks(h.M_GRID, [str(m) for m in h.M_GRID])
        bottom.axhline(1.0, color=REFERENCE_COLOUR, lw=1.2, ls="--")
        bottom.axhline(0.0, color=REFERENCE_COLOUR, lw=1.2, ls="--")
    # One y-range for every sigma_x, wide enough for m = 1 at sigma_x = 1,
    # where LESS-B is worse than OLS.
    for ax in axes[1]:
        ax.set_ylim(-0.05, max(1.15, rel_max + 0.05))
    axes[0, 0].set_ylabel("Test MSE")
    axes[1, 0].set_ylabel("Relative excess risk")
    fig.suptitle(VARIANT_LABELS.get(method, method))

    handles = _kappa_handles(kappas) + [
        Line2D([], [], color=REFERENCE_COLOUR, marker="s", ms=8, ls="", label="OLS"),
        Line2D([], [], color=REFERENCE_COLOUR, marker="_", ms=16, mew=3, ls="", label="Bayes"),
    ]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=SMALL_SIZE,
               handlelength=1.5, columnspacing=1.2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.01))
    _save(fig, f"mse_vs_n_subsets_{method}.png")


def plot_selected(table):
    """Test MSE / OLS test MSE at the cross-validated n_subsets."""
    methods = [m for m in VARIANT_LABELS if m in set(table.method)]
    kappas = sorted(table.kappa.unique())
    separation = h.A_SCALE / np.array(h.SIGMA_X_GRID)

    fig, axes = plt.subplots(1, len(methods), figsize=(0.85 * FIG_WIDTH, 6.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods, strict=True):
        for colour, k in zip(KAPPA_COLOURS, kappas, strict=False):
            t = table[(table.method == method) & (table.kappa == k)].sort_values("sigma_x")
            ax.errorbar(
                h.A_SCALE / t.sigma_x.to_numpy(), t.ratio, yerr=t.ratio_ci,
                color=colour, lw=2.5, marker="o", ms=7, capsize=4,
            )
        ax.axhline(1.0, color=REFERENCE_COLOUR, lw=1.2, ls="--")
        ax.set_xticks(separation, [f"{v:.3g}" for v in separation])
        ax.set_xlabel(r"Cluster separation ($a/\sigma_x$)")
        ax.set_title(VARIANT_LABELS[method])
        ax.set_ylim(0.0, 1.2)
        _grid(ax)
    axes[0].set_ylabel("Test MSE / OLS test MSE")

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=_kappa_handles(kappas), loc="lower center", ncol=len(kappas),
               fontsize=SMALL_SIZE, handlelength=1.5, columnspacing=1.2,
               fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))
    _save(fig, "selected_n_subsets.png")


# Small instance used by the example figures.
EX_CLUSTERS, EX_FEATURES, EX_N, EX_SEED = 2, 5, 500, 3


def _example(kappa, sigma_x, n_test=0):
    rng = np.random.default_rng(EX_SEED)
    centres, theta = h.make_params(EX_CLUSTERS, EX_FEATURES, kappa, h.A_SCALE, rng)
    noise_std = h.NOISE_RATIO * sigma_x
    train = h.sample(centres, theta, EX_N, sigma_x, noise_std, rng)
    test = h.sample(centres, theta, n_test, sigma_x, noise_std, rng) if n_test else None
    return centres, theta, train, test


def _scatter(ax, x, y, labels):
    for j, colour in enumerate(CLUSTER_COLOURS):
        m = labels == j
        ax.scatter(x[m], y[m], s=14, alpha=0.6, color=colour, edgecolors="white",
                   linewidths=0.3)


def _cluster_legend(fig, axes, bottom):
    handles = [
        Line2D([], [], color=c, marker="o", ms=9, ls="", label=f"Cluster {j + 1}")
        for j, c in enumerate(CLUSTER_COLOURS)
    ]
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))


def plot_example_data():
    """One row per cluster separation: the inputs projected on the unit vectors
    of the two centres, then y against <theta_1, x> for three kappa."""
    kappas = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(2, 1 + len(kappas), figsize=(FIG_WIDTH, 10.5))
    for row, s in zip(axes, [1.0, 0.2], strict=True):
        centres, _, (X, _, labels), _ = _example(0.0, s)
        U = centres / np.linalg.norm(centres, axis=0)
        P = X @ U
        _scatter(row[0], P[:, 0], P[:, 1], labels)
        for j, colour in enumerate(CLUSTER_COLOURS):
            row[0].scatter(*(centres[:, j] @ U), s=220, marker="X", color=colour,
                           edgecolors="black", linewidths=1.2, zorder=3)
        row[0].set_aspect("equal", adjustable="datalim")
        row[0].set_title(rf"Inputs, $a/\sigma_x = {h.A_SCALE / s:g}$")
        # ||c_j|| = a = 1, so the coordinates are <c_1, x> and <c_2, x>.
        row[0].set_xlabel(r"$\langle c_1, x \rangle$")
        row[0].set_ylabel(r"$\langle c_2, x \rangle$")
        for ax, k in zip(row[1:], kappas, strict=True):
            _, theta, (X, y, labels), _ = _example(k, s)
            _scatter(ax, X @ theta[:, 0], y, labels)
            ax.set_title(rf"Response, $\kappa = {k:g}$")
            ax.set_xlabel(r"$\langle \theta_1, x \rangle$")
            ax.set_ylabel(r"$y$")
        for ax in row:
            _grid(ax)
    _cluster_legend(fig, axes, 0.06)
    _save(fig, "example_data.png")


def plot_example_fit():
    """True against predicted y on an independent test set (kappa = 0,
    a/sigma_x = 5)."""
    _, _, (X, y, _), (X_te, y_te, labels_te) = _example(0.0, 0.2, n_test=1000)
    models = [
        ("OLS", h.scaled(LinearRegression())),
        (rf"{VARIANT_LABELS['LESS-B-noG']}, $m = 20$", h.less("LESS-B-noG", 20)),
        (rf"{VARIANT_LABELS['LESS-B']}, $m = 20$", h.less("LESS-B", 20)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 7.4), sharex=True, sharey=True)
    lo, hi = y_te.min(), y_te.max()
    for ax, (name, model) in zip(axes, models, strict=True):
        pred = model.fit(X, y).predict(X_te)
        _scatter(ax, y_te, pred, labels_te)
        ax.plot([lo, hi], [lo, hi], color=REFERENCE_COLOUR, lw=1.2, ls="--")
        ax.set_title(f"{name}\nTest MSE = {np.mean((y_te - pred) ** 2):.4f}")
        ax.set_xlabel(r"True $y$")
        _grid(ax)
    axes[0].set_ylabel(r"Predicted $y$")
    _cluster_legend(fig, axes, 0.1)
    _save(fig, "example_fit.png")


def main():
    df = pd.read_csv(os.path.join(h.OUT_DIR, "results.csv"))
    for method in VARIANT_LABELS:
        if method in set(df.method):
            plot_mse_vs_n_subsets(df, method)
    path = os.path.join(h.OUT_DIR, "selected_summary.csv")
    if os.path.exists(path):
        plot_selected(pd.read_csv(path))
    plot_example_data()
    plot_example_fit()


if __name__ == "__main__":
    main()
