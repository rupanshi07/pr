"""
Shared helpers for evaluating the dataset-driven (UCI) Bayesian Networks.

Everything here is about doing the evaluation *properly*:

* Discretization uses fixed, rule-based cut-points (the same rules as
  dataset_bayesian_network.py). Nothing is fitted to the data, so there is
  no preprocessing leakage across cross-validation folds.
* G1 and G2 (earlier period grades) are deliberately NOT used as predictors:
  G3 is almost a function of them, which would make the task trivial.
* All resampling / prior estimation / recalibration is fitted on the
  TRAINING part of each fold only.
* The positive class for every metric is Fail (the minority, ~15%), so
  PR-AUC, recall, F1 and MCC describe the class we actually care about.
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)

DATA_PATH = "data/student_performance_raw.csv"
ESS = 10  # BDeu equivalent sample size, same as dataset_bayesian_network.py

STATE_NAMES = {
    "StudyTime": ["Low", "Medium", "High"],
    "Failures": ["None", "One_or_more"],
    "Absences": ["Low", "Medium", "High"],
    "Medu": ["Low", "High"],
    "Higher": ["No", "Yes"],
    "Pass": ["No", "Yes"],
}

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def load_discretized(path=DATA_PATH):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["Pass"] = np.where(df["G3"] >= 10, "Yes", "No")
    out["StudyTime"] = df["studytime"].apply(
        lambda v: "Low" if v <= 1 else ("Medium" if v == 2 else "High"))
    out["Failures"] = df["failures"].apply(lambda v: "None" if v == 0 else "One_or_more")
    out["Absences"] = df["absences"].apply(
        lambda v: "Low" if v <= 4 else ("Medium" if v <= 10 else "High"))
    # Mother's education: 0-2 = up to 9th grade, 3-4 = secondary / higher
    out["Medu"] = df["Medu"].apply(lambda v: "Low" if v <= 2 else "High")
    # Student wants to take higher education
    out["Higher"] = df["higher"].astype(str).str.strip().str.lower().map({"yes": "Yes", "no": "No"})
    return out


# ----------------------------------------------------------------------------
# Structures
# ----------------------------------------------------------------------------

STRUCTURES = {
    # The network currently in dataset_bayesian_network.py (3 parents -> Pass)
    "orig3": [("StudyTime", "Pass"), ("Failures", "Pass"), ("Absences", "Pass")],

    # Naive Bayes: Pass is the single parent of every observed feature
    "naive": [("Pass", "StudyTime"), ("Pass", "Failures"), ("Pass", "Absences"),
              ("Pass", "Medu"), ("Pass", "Higher")],

    # Flat "converging" DAG: all five features are direct parents of Pass.
    # No hidden layering, but the Pass CPT has 3*2*3*2*2 = 72 parent columns.
    "flat5": [("StudyTime", "Pass"), ("Failures", "Pass"), ("Absences", "Pass"),
              ("Medu", "Pass"), ("Higher", "Pass")],

    # Multi-layer DAG: background -> aspiration -> behaviour -> outcome.
    #   Medu -> Higher                      (family background shapes aspiration)
    #   Medu, Higher -> StudyTime           (aspiration/background shape study habits)
    #   Higher -> Failures, Absences        (aspiration shapes past failures / attendance)
    #   StudyTime, Failures, Absences -> Pass  (behaviour determines the outcome)
    # Pass has only 3*2*3 = 18 parent columns, i.e. a much smaller CPT than flat5.
    "layered": [("Medu", "Higher"), ("Medu", "StudyTime"), ("Higher", "StudyTime"),
                ("Higher", "Failures"), ("Higher", "Absences"),
                ("StudyTime", "Pass"), ("Failures", "Pass"), ("Absences", "Pass")],
}

# Which variables are observed at prediction time in each scenario
SCENARIOS = {
    # All behavioural + background variables known (end of term)
    "full": ["StudyTime", "Failures", "Absences", "Medu", "Higher"],
    # Early warning: only background variables known; behaviour not yet observed.
    "early": ["Medu", "Higher"],
}


def structure_nodes(name):
    nodes = set()
    for a, b in STRUCTURES[name]:
        nodes.update([a, b])
    return nodes


def fit_structure(name, train_df):
    nodes = sorted(structure_nodes(name))
    model = DiscreteBayesianNetwork(STRUCTURES[name])
    estimator = BayesianEstimator(
        model, train_df[nodes],
        state_names={n: STATE_NAMES[n] for n in nodes},
    )
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=ESS)
    model.add_cpds(*cpds)
    assert model.check_model()
    return model


def predict_p_pass(model, test_df, observed):
    """P(Pass=Yes | evidence) for each row, using exact inference.
    Evidence is restricted to `observed` variables that exist in the model.
    Results are cached per distinct evidence tuple (at most 72 of them)."""
    infer = VariableElimination(model)
    usable = [v for v in observed if v in model.nodes() and v != "Pass"]
    if not usable:
        # Model has none of the observed variables (e.g. orig3 in the "early"
        # scenario): its prediction is just the marginal P(Pass), for everyone.
        res = infer.query(["Pass"], show_progress=False)
        return np.full(len(test_df), res.get_value(Pass="Yes"))
    cache = {}
    out = np.full(len(test_df), np.nan)
    rows = list(test_df[usable].itertuples(index=False, name=None))
    for i, row in enumerate(rows):
        if row not in cache:
            ev = dict(zip(usable, row))
            res = infer.query(["Pass"], evidence=ev, show_progress=False)
            cache[row] = res.get_value(Pass="Yes")
        out[i] = cache[row]
    assert not np.isnan(out).any()
    return out


# ----------------------------------------------------------------------------
# Class-imbalance handling (always applied to the training fold only)
# ----------------------------------------------------------------------------


def resample_train(train_df, strategy, rng):
    """strategy: 'none' | 'oversample' | 'undersample'. Returns resampled df
    whose Pass classes are balanced 50/50 (unless 'none')."""
    if strategy == "none":
        return train_df
    pos = train_df[train_df["Pass"] == "Yes"]
    neg = train_df[train_df["Pass"] == "No"]
    major, minor = (pos, neg) if len(pos) >= len(neg) else (neg, pos)
    if strategy == "oversample":
        extra = minor.sample(n=len(major) - len(minor), replace=True,
                             random_state=int(rng.integers(1 << 31)))
        return pd.concat([major, minor, extra], ignore_index=True)
    if strategy == "undersample":
        keep = major.sample(n=len(minor), replace=False,
                            random_state=int(rng.integers(1 << 31)))
        return pd.concat([keep, minor], ignore_index=True)
    raise ValueError(strategy)


def prior_correct(p, train_prior, sample_prior):
    """Bayes prior-shift correction. A model trained on data whose class
    prior was `sample_prior` is converted to the true prior `train_prior`
    (both are P(class of p)). Works elementwise on arrays p = P(class)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = p / (1 - p)
    odds *= (train_prior / (1 - train_prior)) / (sample_prior / (1 - sample_prior))
    return odds / (1 + odds)


# ----------------------------------------------------------------------------
# Calibration utilities
# ----------------------------------------------------------------------------


def calibration_bins(p, y, n_bins=10):
    """Equal-width reliability bins. Returns arrays (mean_pred, frac_pos, count)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)
    mp, fp, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        cnt.append(int(m.sum()))
        mp.append(p[m].mean() if m.any() else np.nan)
        fp.append(y[m].mean() if m.any() else np.nan)
    return np.array(mp), np.array(fp), np.array(cnt)


def ece(p, y, n_bins=10):
    mp, fp, cnt = calibration_bins(p, y, n_bins)
    ok = cnt > 0
    return float(np.sum(cnt[ok] * np.abs(mp[ok] - fp[ok])) / cnt.sum())


def fit_platt(p, y):
    """Logistic recalibration on logit(p). Returns (a, b) with p' = sigmoid(a*logit(p)+b)."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, a, b):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(a * z + b)))


# ----------------------------------------------------------------------------
# Metrics (positive class = Fail)
# ----------------------------------------------------------------------------


def probability_metrics(p_fail, y_fail):
    p = np.clip(p_fail, 1e-6, 1 - 1e-6)
    out = {
        "brier": float(np.mean((p - y_fail) ** 2)),
        "logloss": float(log_loss(y_fail, p, labels=[0, 1])),
        "ece": ece(p, y_fail),
    }
    if len(np.unique(y_fail)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_fail, p))
        out["pr_auc"] = float(average_precision_score(y_fail, p))
    return out


def decision_metrics(p_fail, y_fail, threshold):
    pred = (p_fail >= threshold).astype(int)
    return {
        "bal_acc": float(balanced_accuracy_score(y_fail, pred)),
        "recall_fail": float(recall_score(y_fail, pred, zero_division=0)),
        "precision_fail": float(precision_score(y_fail, pred, zero_division=0)),
        "f1_fail": float(f1_score(y_fail, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_fail, pred)),
        "pred_fail_rate": float(pred.mean()),
    }
