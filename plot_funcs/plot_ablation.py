"""Plots the ablation results (logs/ablation) to plots/ablation/."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.names import display_name
from src.resultdirs import ABLATION

# Set plot style
sns.set_theme(style='whitegrid', color_codes=True)
sns.set_style('ticks')

# Use the vibrant colors from plot_variants.py
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)



def _complete_problems(result_dir, n_arms=8, n_folds=4):
    """Problems from src.problems whose cells are all there, in suite order.

    A problem is included only when every arm has all of its folds, counting
    cells recorded in _failed; problems with missing cells are printed and
    skipped.
    """
    import json as _json
    import os as _os

    from src.problems import ALL_PROBLEMS

    out, partial = [], []
    for name in ALL_PROBLEMS:
        path = _os.path.join(result_dir, f"{name}.json")
        if not _os.path.exists(path):
            continue
        with open(path) as fh:
            data = _json.load(fh)
        cells = {k: len(v) for k, v in data.items()
                 if not k.startswith("_") and isinstance(v, dict)}
        failed = sum(len(v) for v in (data.get("_failed") or {}).values())
        if len(cells) == n_arms and sum(cells.values()) + failed == n_arms * n_folds:
            # a cell that failed carries its reason in _failed, so it is a
            # recorded failure rather than an unknown gap: the arm is averaged
            # over the folds that did run
            out.append(name)
            if failed:
                print(f"  {name}: {failed} hucre basarisiz, kalan fold'lardan cizildi")
        else:
            partial.append(f"{name} ({sum(cells.values())}/{n_arms * n_folds})")
    if partial:
        print(f"  atlandi (eksik hucre): {', '.join(partial)}")
    return out


def _label(name):
    """Axis label: the paper name of the problem."""
    return display_name(name)


def load_results(save_dir=ABLATION):
    """Load all JSON results from the save directory"""
    results = {}

    problem_names = _complete_problems(save_dir)

    for problem_name in problem_names:
        json_path = os.path.join(save_dir, f"{problem_name}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                results[problem_name] = json.load(f)
        else:
            print(f"Warning: {json_path} not found!")
    
    return results

def extract_mean_scores(results):
    """Extract mean MSE scores for each method and problem"""
    av_methods = ['AV-NoW-NoG', 'AV-W-NoG', 'AV-NoW-G', 'AV-N-G']
    gb_methods = ['B-NoW-NoG', 'B-W-NoG', 'B-NoW-G', 'B-N-G']
    
    av_data = []
    gb_data = []
    problem_names = []
    
    for problem_name, problem_results in sorted(results.items()):
        problem_names.append(problem_name)
        
        # Extract AV results
        av_scores = []
        for method in av_methods:
            if method in problem_results:
                fold_scores = [float(score) for score in problem_results[method].values()]
                av_scores.append(np.mean(fold_scores))
            else:
                av_scores.append(np.nan)
        av_data.append(av_scores)
        
        # Extract GB results
        gb_scores = []
        for method in gb_methods:
            if method in problem_results:
                fold_scores = [float(score) for score in problem_results[method].values()]
                gb_scores.append(np.mean(fold_scores))
            else:
                gb_scores.append(np.nan)
        gb_data.append(gb_scores)
    
    return np.array(av_data), np.array(gb_data), problem_names

# Thresholds on the ratio of the worst arm to full LESS, used by split_by_spread.
DIVERGING_RATIO = 10.0
EXTREME_RATIO = 1e10


def split_by_spread(data):
    """Index lists by how far the worst arm is from full LESS: calm, wide, extreme."""
    ratio = data / data[:, -1].reshape(-1, 1)
    worst = np.nanmax(np.nan_to_num(ratio, nan=1.0, posinf=np.inf), axis=1)
    calm = [i for i, w in enumerate(worst) if w <= DIVERGING_RATIO]
    wide = [i for i, w in enumerate(worst) if DIVERGING_RATIO < w <= EXTREME_RATIO]
    extreme = [i for i, w in enumerate(worst) if w > EXTREME_RATIO]
    return calm, wide, extreme


def create_ablation_plot(data, problem_names, title_suffix, method_labels):
    """Create ablation study bar plot"""
    # Scaled by the row maximum and drawn on a linear axis. Bars below
    # TOO_SMALL are annotated with their value so they remain readable.
    data_normalized = data / np.max(data, axis=1).reshape(-1, 1)
    data_normalized = np.nan_to_num(data_normalized, nan=1.0)

    fig, ax = plt.subplots(figsize=(max(10.0, 0.8 * len(problem_names) + 3), 5.5))
    n_methods = len(method_labels)
    width = 0.8 / n_methods
    x = np.arange(len(problem_names)) * 1.2

    TOO_SMALL = 0.02
    for i, label in enumerate(method_labels):
        position = x - (n_methods / 2 - i) * width
        ax.bar(position, data_normalized[:, i], width * 0.8, bottom=0,
               color=bar_colors[i], label=label)
        for j, v in enumerate(data_normalized[:, i]):
            if 0 < v < TOO_SMALL:
                txt = f"{v:.0e}".replace("e-0", "e-")
                ax.text(position[j], 0.03, txt, ha='center', va='bottom',
                        fontsize=6.5, rotation=90, color=bar_colors[i])

    ax.set_ylabel('Scaled MSE')
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(p) for p in problem_names], rotation=45, ha='right')
    ax.set_title(f'Ablation Study Results - {title_suffix}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.42),
              fancybox=True, shadow=True, ncol=4)

    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    fig.tight_layout()

    return fig, ax

def plot_ablation_results(load_dir=ABLATION, save_dir="plots/ablation", save_plots=True):
    """Main function to create ablation plots"""
    # Load results
    results = load_results(load_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    if not results:
        print("No results found! Make sure the JSON files exist in the save directory.")
        return
    
    # Extract mean scores
    av_data, gb_data, problem_names = extract_mean_scores(results)
    
    # Method labels, in the order of the methods in extract_mean_scores
    av_labels = ['NoW-NoG', 'W-NoG', 'NoW-G', 'LESS (W-G)']
    gb_labels = ['NoW-NoG', 'W-NoG', 'NoW-G', 'LESS (W-G)']
    
    for data, labels, tag, title in [(av_data, av_labels, 'av', 'LESS-A'),
                                     (gb_data, gb_labels, 'gb', 'LESS-B')]:
        if data.size == 0:
            continue
        fig, _ax = create_ablation_plot(data, problem_names, title, labels)
        if save_plots:
            fig.savefig(os.path.join(save_dir, f'ablation_{tag}_results.png'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  {title}: {len(problem_names)} veri tek sekilde")

if __name__ == "__main__":
    # Run the plotting function
    plot_ablation_results(save_plots=True)