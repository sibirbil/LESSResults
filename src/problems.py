"""The list of problems every experiment runs on, with the fold policy of each.

Each entry maps a problem name to (loader, group spec). The group spec says how
cross-validation folds are cut:

  None          rows are independent; shuffled KFold.
  <loader>      a group id per row (patient, day, post, ...); GroupKFold on it.
  ROW_IDENTITY  identical feature rows repeat; the row itself is the group, so a
                repeated record never lands on both sides of a split.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.datasets import (
    abalone,
    airfoil,
    airquality,
    airquality_groups,
    blogfeedback,
    blogfeedback_groups,
    cadata,
    casp,
    ccpp,
    cpusmallscale,
    crime,
    ctslices,
    ctslices_groups,
    edwards,
    edwards_groups,
    energy,
    housing,
    lasrosas,
    metro,
    metro_groups,
    sgemm,
    srdb,
    srdb_groups,
    twitter,
)

# Sentinel for "the group is the feature row itself"; see the module docstring.
ROW_IDENTITY = "row_identity"


def row_identity_groups(X):
    """Group id per row, equal for rows whose features are identical.

    Hashing the feature matrix recovers the repeated record when the dataset
    ships no id of its own. Rows that merely collide would be merged, which errs
    towards fewer leaks rather than more.
    """
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values


def merge_group_labels(*labelings):
    """One label per row, merging rows that share a label in ANY of the inputs.

    Two dependencies can each be captured by their own id and still leak
    together. srdb is the case: a site repeats across studies, so grouping by
    site or by study alone fixes half of it; what has to stay together is the
    connected component of the two relations, which is what this returns -- a
    union-find over the rows, joining each labeling's members in turn
    (script_funcs/prepare_hetero_datasets.py).
    """
    labelings = [np.asarray(lab) for lab in labelings]
    n = len(labelings[0])
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lab in labelings:
        order = np.argsort(lab, kind="stable")
        first = order[0]
        for prev, cur in itertools.pairwise(order):
            if lab[cur] == lab[prev]:
                a, b = find(first), find(cur)
                if a != b:
                    parent[b] = a
            else:
                first = cur
    return np.array([find(i) for i in range(n)])


SMALL_PROBLEMS = {
    f.__name__.upper(): (f, None) for f in
    (abalone, airfoil, cadata, cpusmallscale, energy, housing)
}
# ccpp ships 41 pairs of identical records -- same features, same target. UCI
# publishes the file already shuffled, so no original ordering survives to group
# by; the repeated row itself is the group id, which stops a test row from also
# being a training row.
SMALL_PROBLEMS["CCPP"] = (ccpp, ROW_IDENTITY)
# blogfeedback ships an official train/test split, which these runners do not
# use; like every other problem it is scored by cross-validation, grouped by
# post identity so the same blog does not land on both sides of a split.
HETERO_PROBLEMS = {
    # ctslices has many slices per patient, so folds are grouped by patient.
    "CTSLICES_GROUPED": (ctslices, ctslices_groups),
    "SGEMM": (sgemm, None),
    "CASP": (casp, ROW_IDENTITY),
    "CRIME": (crime, None),
    "BLOGFEEDBACK": (blogfeedback, blogfeedback_groups),
    "TWITTER": (twitter, ROW_IDENTITY),
    "METRO": (metro, metro_groups),
}
# Problems that also carry an environment label. For lasrosas the label
# (landscape position) lives in datasetsR/lasrosas_env.csv and is not the fold
# policy.
LOCAL_PROBLEMS = {
    "LASROSAS": (lasrosas, None),
    # edwards repeats every (environment, genotype) cell across three replicate
    # blocks and airquality is one uninterrupted hourly series, so both carry a
    # fold policy of their own -- the cell and the day. Neither is the
    # environment: those live in datasetsR/*_env.csv and never decide a split.
    "EDWARDS": (edwards, edwards_groups),
    "AIRQUALITY": (airquality, airquality_groups),
    # srdb repeats a site across studies -- one coordinate appears 130 times --
    # so the fold unit is the connected component of site and study
    # (merge_group_labels above).
    "SRDB": (srdb, srdb_groups),
}
ALL_PROBLEMS = {**SMALL_PROBLEMS, **HETERO_PROBLEMS, **LOCAL_PROBLEMS}
