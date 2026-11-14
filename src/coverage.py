"""Method-problem cells of the main comparison that are skipped.

main.py and script_funcs/anfis_test.py consult EXCLUDED before fitting and
write the reason under "_skipped" in the problem's result file, one entry per
outer fold. Every other cell is fitted.
"""

N_FOLDS = 4

EXCLUDED = {
    # ANFIS: scikit-anfis builds 2^p rules, one per combination of the two
    # membership functions it places on each feature, and its cost grows with
    # rows x rules.
    ("ANFIS", "BLOGFEEDBACK"): "2^280 rules for 280 features",
    ("ANFIS", "CRIME"): "2^100 rules for 100 features",
    ("ANFIS", "CTSLICES_GROUPED"): "2^384 rules for 384 features",
    ("ANFIS", "EDWARDS"): "2^87 rules for 87 features",
    ("ANFIS", "ENERGY"): "2^26 rules for 26 features",
    ("ANFIS", "METRO"): "2^67 rules for 67 features",
    ("ANFIS", "TWITTER"): "2^77 rules for 77 features",
    ("ANFIS", "CASP"): "rows x rules = 45,730 x 2^9 = 2.3e7",
    ("ANFIS", "CPUSMALLSCALE"): "rows x rules = 8,192 x 2^12 = 3.4e7",
    ("ANFIS", "SGEMM"): "rows x rules = 241,600 x 2^14 = 4.0e9",
    # GPR: the kernel matrix and its Cholesky factor are two n x n float64
    # arrays.
    ("GPR", "BLOGFEEDBACK"): "2 x 28.8 GB for the n x n kernel and its factor at n = 60,021",
    ("GPR", "CASP"): "2 x 16.7 GB for the n x n kernel and its factor at n = 45,730",
    ("GPR", "CTSLICES_GROUPED"): "2 x 22.9 GB for the n x n kernel and its factor at n = 53,500",
    ("GPR", "METRO"): "2 x 18.6 GB for the n x n kernel and its factor at n = 48,204",
    ("GPR", "SGEMM"): "2 x 467 GB for the n x n kernel and its factor at n = 241,600",
    ("GPR", "TWITTER"): "2 x 2,721 GB for the n x n kernel and its factor at n = 583,250",
    # LOCR: one weighted least-squares fit per test point over frac * n
    # neighbours.
    ("LOCR", "BLOGFEEDBACK"): "one weighted fit per test point, n = 60,021, p = 280",
    ("LOCR", "CTSLICES_GROUPED"): "one weighted fit per test point, n = 53,500, p = 384",
    ("LOCR", "METRO"): "one weighted fit per test point, n = 48,204, p = 67",
    ("LOCR", "TWITTER"): "one weighted fit per test point, n = 583,250, p = 77",
    # TSK: fuzzy c-means and one rule set per grid candidate.
    ("TSK", "BLOGFEEDBACK"): "fuzzy c-means per candidate, n = 60,021, p = 280",
    ("TSK", "CTSLICES_GROUPED"): "fuzzy c-means per candidate, n = 53,500, p = 384",
    ("TSK", "SGEMM"): "fuzzy c-means per candidate, n = 241,600, p = 14",
    ("TSK", "TWITTER"): "fuzzy c-means per candidate, n = 583,250, p = 77",
}


def excluded(method, problem):
    """Reason the cell is skipped, or None."""
    return EXCLUDED.get((method, problem))


def skip_entries(method, problem):
    """The "_skipped" record of an excluded cell, keyed by fold."""
    reason = EXCLUDED[(method, problem)]
    return {str(k): {"reason": reason} for k in range(1, N_FOLDS + 1)}
