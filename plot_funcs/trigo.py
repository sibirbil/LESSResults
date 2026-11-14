"""Fits LESS-A and LESS-B on a synthetic trigonometric dataset and plots the
fitted curves for several settings to plots/trigo/.

    PYTHONPATH=. python plot_funcs/trigo.py
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from less import LESSARegressor, LESSBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE

plt.rcParams["mathtext.fontset"] = "cm"  # "cm" = Computer Modern
plt.rcParams["font.family"] = "serif"

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "figure.titlesize": 24,
    }
)

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Parameter combinations
PARAM_COMBINATIONS = [
    {"n_subsets": 5, "min_neighbors": 40},
    {"n_subsets": 10, "min_neighbors": 20},
    {"n_subsets": 20, "min_neighbors": 10},
    {"n_subsets": 50, "min_neighbors": 4},
]

N_ESTIMATORS_LIST = [5, 10, 20]


def generate_trigonometric_dataset(n_samples=1000, noise_std=0.1, random_state=None):
    """
    Synthetic dataset: a trigonometric function plus Gaussian noise.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.uniform(0, 4 * np.pi, n_samples).reshape(-1, 1)
    y_true = np.sin(2 * X.flatten()) + 0.3 * np.sin(4 * X.flatten()) + 0.2 * np.cos(3 * X.flatten())
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    return X, y


def run_experiment(save_dir="results"):
    """
    Runs the LESS regressor experiments on the trigonometric dataset.

    Args:
        save_dir (str): Directory the results are saved to
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Generating trigonometric dataset...")
    X, y = generate_trigonometric_dataset(n_samples=1000, noise_std=0.1, random_state=RANDOM_STATE)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Build and fit the model
    plot_model = LESSARegressor(
        random_state=RANDOM_STATE,
        local_estimator="linear",
        global_estimator=LinearRegression,
        n_estimators=1,
    )
    plot_model.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.flatten(), y_train, c="blue", alpha=0.6, label="Train", s=20)
    plt.scatter(X_test.flatten(), y_test, c="orange", alpha=0.8, label="Test", s=30)

    # Neighbourhoods of the local models are looked up in X
    X_used = X
    tree = KDTree(X_used)

    # Each local model is drawn as a black line with no legend entry
    for local_model in plot_model._local_models_iterations[0]:
        estimator = local_model.estimator
        center = local_model.center

        # Neighbours of this center
        n_neighbors = plot_model._n_neighbors
        _, neighbor_indices = tree.query([np.reshape(center, (-1,))], k=n_neighbors)
        neighbor_indices = neighbor_indices[0]

        # X range of the neighbours
        X_neighbors = X_used[neighbor_indices].flatten()
        x_min, x_max = X_neighbors.min(), X_neighbors.max()

        # Local linear prediction over this range
        X_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_line = estimator.predict(X_line)
        plt.plot(X_line.flatten(), y_line, color="black", linewidth=2.0, alpha=0.9)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()  # only Train/Test appear
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    data_plot_path = os.path.join(save_dir, "data_visualization.png")
    plt.savefig(data_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Data visualization saved to {data_plot_path}")

    result_path = os.path.join(save_dir, "TRIGONOMETRIC.json")

    # Load existing results
    if os.path.exists(result_path):
        with open(result_path) as f:
            previous_results = json.load(f)
    else:
        previous_results = {}

    # Predictions per setting
    av_predictions = {}  # {(n_subsets, min_neighbors): {n_estimators: y_pred}}
    gb_predictions = {}

    # Sort X_test for plotting
    sorted_indices = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sorted_indices]

    # For each parameter combination
    for param_combo in PARAM_COMBINATIONS:
        n_subsets = param_combo["n_subsets"]
        min_neighbors = param_combo["min_neighbors"]
        param_key = (n_subsets, min_neighbors)

        av_predictions[param_key] = {}
        gb_predictions[param_key] = {}

        # For each n_estimators value
        for n_estimators in N_ESTIMATORS_LIST:
            models = [
                (
                    LESSARegressor(
                        random_state=RANDOM_STATE,
                        local_estimator="linear",
                        global_estimator=LinearRegression,
                        n_subsets=n_subsets,
                        n_estimators=n_estimators,
                        min_neighbors=min_neighbors,
                    ),
                    f"LESSARegressor_est{n_estimators}_sub{n_subsets}_min{min_neighbors}",
                    "AV",
                ),
                (
                    LESSBRegressor(
                        random_state=RANDOM_STATE,
                        local_estimator="linear",
                        global_estimator=LinearRegression,
                        n_subsets=n_subsets,
                        n_estimators=n_estimators,
                        min_neighbors=min_neighbors,
                    ),
                    f"LESSBRegressor_est{n_estimators}_sub{n_subsets}_min{min_neighbors}",
                    "GB",
                ),
            ]

            for estimator, estimator_type, model_class in models:
                # Skip if already computed
                if estimator_type in previous_results:
                    print(f"Skipping {estimator_type} (already computed).")
                    continue

                print(f"Training {estimator_type}...")

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

                # Fit the model
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)

                print(f"{estimator_type} MSE: {mse:.4f}")

                # Store predictions in sorted order
                y_pred_sorted = y_pred[sorted_indices]
                if model_class == "AV":
                    av_predictions[param_key][n_estimators] = y_pred_sorted
                else:
                    gb_predictions[param_key][n_estimators] = y_pred_sorted

    # One plot per setting for the AV predictions
    print("Generating separate plots for LESSARegressor predictions...")
    av_palette = sns.color_palette("hls", 12)
    plot_index = 0
    for param_combo in PARAM_COMBINATIONS:
        for n_est in N_ESTIMATORS_LIST:
            n_subsets = param_combo["n_subsets"]
            min_neighbors = param_combo["min_neighbors"]
            param_key = (n_subsets, min_neighbors)

            if param_key in av_predictions and n_est in av_predictions[param_key]:
                y_pred = av_predictions[param_key][n_est]
                color = av_palette[plot_index % 12]
                plot_index += 1

                plt.figure(figsize=(10, 6))
                plt.plot(X_test_sorted.flatten(), y_pred, color=color, linewidth=2)
                plt.grid(False)
                plt.tick_params(
                    axis="both",
                    which="both",
                    bottom=True,
                    top=False,
                    left=True,
                    right=False,
                    labelbottom=False,
                    labelleft=False,
                )

                plot_filename = f"LESSARegressor_sub{n_subsets}_min{min_neighbors}_est{n_est}.png"
                av_plot_path = os.path.join(save_dir, plot_filename)
                plt.savefig(av_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"AV prediction plot saved to {av_plot_path}")

    # One plot per setting for the GB predictions
    print("Generating separate plots for LESSBRegressor predictions...")
    gb_palette = sns.color_palette("husl", 12)
    plot_index = 0
    for param_combo in PARAM_COMBINATIONS:
        for n_est in N_ESTIMATORS_LIST:
            n_subsets = param_combo["n_subsets"]
            min_neighbors = param_combo["min_neighbors"]
            param_key = (n_subsets, min_neighbors)

            if param_key in gb_predictions and n_est in gb_predictions[param_key]:
                y_pred = gb_predictions[param_key][n_est]
                color = gb_palette[plot_index % 12]
                plot_index += 1

                plt.figure(figsize=(10, 6))
                plt.plot(X_test_sorted.flatten(), y_pred, color=color, linewidth=2)
                plt.grid(False)
                plt.tick_params(
                    axis="both",
                    which="both",
                    bottom=True,
                    top=False,
                    left=True,
                    right=False,
                    labelbottom=False,
                    labelleft=False,
                )

                plot_filename = f"LESSBRegressor_sub{n_subsets}_min{min_neighbors}_est{n_est}.png"
                gb_plot_path = os.path.join(save_dir, plot_filename)
                plt.savefig(gb_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"GB prediction plot saved to {gb_plot_path}")

    print("All experiments completed!")

    # Print a summary of the results
    print("\n=== RESULTS SUMMARY ===")
    for model_name, mse in sorted(previous_results.items(), key=lambda x: x[1]):
        print(f"{model_name}: MSE = {mse:.4f}")


if __name__ == "__main__":
    run_experiment(save_dir="plots/trigo")
