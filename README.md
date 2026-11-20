# LESS Results

The computational study for our method Learning with Subset Stacking (LESS).

`pip install less-learn`

## Data

The datasets are not stored in the repository. Download `datasetsR.zip` from the
link below and extract it so that the CSV files sit in a `datasetsR/` folder at
the repository root:

https://1drv.ms/u/c/0b26aa4e19835550/IQCEN0mqxK_3TI87u5uZ_jvoAQj8fQPWWK9t2MeiRfxtcak?e=rkVz94

```
LESSResults/
├── datasetsR/
│   ├── abalone.csv
│   ├── ...
├── main.py
└── ...
```

## Results

The results are stored in `logs/`, one JSON file per problem except for the
synthetic heterogeneity experiment:

| Directory | Study |
|---|---|
| `logs/results/` | comparison against the other methods (main table, rankings, statistical tests) |
| `logs/ablation/` | ablation study |
| `logs/incremental/` | local and global learners |
| `logs/variants/` | subset selection and data splitting |
| `logs/timing/` | computation times |
| `logs/mag_env/` | Magging with the environments of the data |
| `logs/heterogeneity/` | synthetic data with controlled heterogeneity (`results.csv`: test MSE for every `n_subsets`; `cv.csv` and `selected_summary.csv`: `n_subsets` chosen by 4-fold cross-validation) |

In the cross-validated studies, method keys hold the test MSE of the four
outer folds, and the configuration the inner cross-validation selected on each
fold is stored under `_metrics` (main comparison) or `_best_params` (ANFIS and
the other studies). A method that is skipped on a problem is listed in
`src/coverage.py`, and its reason is written under `_skipped`. The timing files
hold fit and predict times in seconds.

## Reproducing the figures and tables

Install the dependencies with `pip install -r requirements.txt` and run every
script from the repository root with `PYTHONPATH=.`.

From the stored results:

```
PYTHONPATH=. python plot_funcs/plot.py               # algorithm rankings
PYTHONPATH=. python plot_funcs/plot_ablation.py      # ablation study
PYTHONPATH=. python plot_funcs/plot_incremental.py   # local and global learners
PYTHONPATH=. python plot_funcs/plot_variants.py      # LESS variants
PYTHONPATH=. python plot_funcs/timing_plot.py        # computation times
PYTHONPATH=. python plot_funcs/plot_heterogeneity.py # synthetic heterogeneity
PYTHONPATH=. python script_funcs/posthoc_table.py    # statistical tests
```

## Rerunning the experiments

The scripts with `--results-dir` resume from the result files in that
directory, so give a new directory to recompute from scratch:

```
PYTHONPATH=. python main.py --no-budget --results-dir rerun
PYTHONPATH=. python script_funcs/anfis_test.py --results-dir rerun
PYTHONPATH=. python script_funcs/ablation.py --results-dir rerun_ablation
PYTHONPATH=. python script_funcs/incremental.py --results-dir rerun_incremental
PYTHONPATH=. python script_funcs/variants.py --results-dir rerun_variants
PYTHONPATH=. python script_funcs/timing.py --datasets energy housing ctslices casp --repeats 2
PYTHONPATH=. python script_funcs/mag_environments.py
PYTHONPATH=. python script_funcs/heterogeneity.py
PYTHONPATH=. python script_funcs/heterogeneity.py --cv
```

`timing.py`, `mag_environments.py` and `heterogeneity.py` write to
`logs/timing/`, `logs/mag_env/` and `logs/heterogeneity/`; move those
directories aside before rerunning them.
`main.py` and `anfis_test.py` run all 18 problems by default and skip the
cells listed in `src/coverage.py`. Run `anfis_test.py` on its own: scikit-anfis
writes a checkpoint with a fixed name to the working directory. The
experiments were run with `less-learn==0.5.0` and the versions pinned in
`requirements.txt`.
