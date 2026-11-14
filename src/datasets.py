"""Loaders for the CSV files in datasetsR/.

Each loader takes the data directory and returns (X, y). The *_groups loaders
return one group id per row, used to keep related rows in the same fold.
"""

import os

import pandas as pd

### REGRESSION

def abalone(wd):
    df = pd.read_csv(os.path.join(wd, "abalone.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def airfoil(wd):
    df = pd.read_csv(os.path.join(wd, "airfoil_self_noise.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def cadata(wd):
    df = pd.read_csv(os.path.join(wd, "cadata.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def ccpp(wd):
    df = pd.read_csv(os.path.join(wd, "ccpp.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def cpusmallscale(wd):
    df = pd.read_csv(os.path.join(wd, "cpusmallscale.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def energy(wd):
    df = pd.read_csv(os.path.join(wd, "energydata_complete.csv"), header=None)
    X = df.iloc[:, 1:].values   # last column is rv1, a random variable that ships with the dataset
    y = df.iloc[:, 0].values    # first column is the target (Appliances)
    return X, y

def housing(wd):
    df = pd.read_csv(os.path.join(wd, "housing.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def lasrosas(wd):
    # Las Rosas corn trial. Anselin et al. (AJAE 2004) report the nitrogen
    # response differing by landscape position.
    df = pd.read_csv(os.path.join(wd, "lasrosas.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def edwards(wd):
    # Iowa State multi-environment oat trial (Edwards & Jannink, Crop Science
    # 2006), studied for its genotype x environment interaction variances.
    # Features are the trial's design plus test weight, so most of the matrix is
    # a one-hot genotype.
    df = pd.read_csv(os.path.join(wd, "edwards.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def airquality(wd):
    # De Vito et al. (2008): a year of hourly readings from one street-level
    # sensor array, calibrating five metal-oxide channels plus temperature and
    # humidity against a reference NOx analyser. The sensor response drifts with
    # season, so the month is the environment.
    df = pd.read_csv(os.path.join(wd, "airquality.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def srdb(wd):
    # Global soil respiration database. Bond-Lamberty & Thomson (Biogeosciences
    # 2010) write that regressing respiration on climate gives results that
    # "vary tremendously across ecosystems and biomes", so the biome is the
    # environment. Five continuous features and no categorical column.
    df = pd.read_csv(os.path.join(wd, "srdb.csv"), header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


### LARGER PROBLEMS AND GROUP IDS
# Header-less CSVs in datasetsR/, features first and the target last.

def _table(wd, filename):
    """Read a header-less CSV into (X, y), the target being the last column."""
    df = pd.read_csv(os.path.join(wd, filename), header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def _groups(wd, filename):
    """Read a one-column CSV of group ids."""
    return pd.read_csv(os.path.join(wd, filename), header=None).iloc[:, 0].values


# --- datasets ---------------------------------------------------------------

def blogfeedback(wd):        return _table(wd, "blogfeedback.csv")
def casp(wd):                return _table(wd, "casp.csv")
def crime(wd):               return _table(wd, "crime.csv")
def ctslices(wd):            return _table(wd, "ctslices.csv")
def metro(wd):               return _table(wd, "metro.csv")
def sgemm(wd):               return _table(wd, "sgemm.csv")
def twitter(wd):             return _table(wd, "twitter.csv")


# --- group ids that ship with the data --------------------------------------

def blogfeedback_groups(wd):     return _groups(wd, "blogfeedback_groups.csv")
def ctslices_groups(wd):         return _groups(wd, "ctslices_groups.csv")
def metro_groups(wd):            return _groups(wd, "metro_groups.csv")
def edwards_groups(wd):          return _groups(wd, "edwards_groups.csv")
def airquality_groups(wd):       return _groups(wd, "airquality_groups.csv")
def srdb_groups(wd):             return _groups(wd, "srdb_groups.csv")
