"""
HMM ablation: does the calibration HMM add predictive information on top of
the Bayesian Network?

Question answered
-----------------
Given everything the live BN already knows, does knowing the learner's
inferred calibration state improve the prediction of whether their NEXT
answer is correct?

Models compared (all stacked on the BN's own P(Correct), all evaluated with
grouped cross-validation so that no learner/session is in both train and test):

  bn           logit(p_bn) only                         <- "without HMM"
  bn+conf      + effective confidence (control: HMM's raw input, no memory)
  bn+hist      + rolling accuracy over the last 5 answers, x confidence
                 (control: a simple non-HMM history feature)
  bn+rule      + rule-based calibration state from hmm_state_definition.py
  bn+hmm       + HMM state belief, x confidence          <- "with HMM"
  bn+oracle    + TRUE hidden state (synthetic data only; upper bound)

IMPORTANT -- no look-ahead
--------------------------
The HMM state belief used for predicting answer t is the FORWARD-FILTERED
predictive distribution using observations 1..t-1 only. (hmmlearn's
decode/predict_proba use the whole sequence, i.e. future answers, which
would leak the label.) The rolling/rule features likewise use only the past.

Two data sources
----------------
  python evaluate_hmm_ablation.py                 # synthetic learners (default)
  python evaluate_hmm_ablation.py --log data/interaction_log.csv   # YOUR real data

Synthetic mode is a MECHANISM CHECK, not evidence: the simulator is built so
that confidence is partly noise for mis-calibrated learners, so a positive
HMM result there only shows the pipeline CAN exploit calibration state when it
exists. The real-log mode is what answers the question for your system.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, STATES
from hmm_state_definition import rolling_calibration_state, CONFIDENCE_NUMERIC

warnings.filterwarnings("ignore")
SEED = 42
LEVELS = ["Low", "Medium", "High"]
DIFFS = ["Easy", "Medium", "Hard"]
DIFF_ACC = {"Easy": 0.75, "Medium": 0.55, "Hard": 0.35}
N_FOLDS, N_REPEATS = 5, 10


# ----------------------------------------------------------------------------
# Synthetic learners with a genuine hidden calibration state
# ----------------------------------------------------------------------------

def make_bn_lookup(bn):
    table = {}
    import itertools
    for conf, diff, t, h, prev in itertools.product(LEVELS, DIFFS, ["Fast", "Slow"],
                                                    ["No", "Yes"], LEVELS):
        table[(conf, diff, t, h, prev)] = estimate_probability_correct(
            bn, confidence=conf, difficulty=diff, time_=t, hints=h, previous_accuracy=prev)
    return table


def simulate(n_learners=200, steps=40, seed=SEED):
    """Hidden state follows the HMM's own transition matrix. Correctness depends
    on difficulty and a learner ability, NOT on the state itself; the state only
    changes what confidence the learner REPORTS (over: too high, under: too low,
    well: tracks true success probability). So a BN that trusts confidence is
    miscalibrated exactly for the mis-calibrated learners."""
    rng = np.random.default_rng(seed)
    hmm_model = build_hmm()
    A = hmm_model.transmat_
    lookup = make_bn_lookup(build_bayesian_network())
    rows = []
    for g in range(n_learners):
        ability = rng.normal(0, 0.8)
        s = rng.choice(3, p=hmm_model.startprob_)
        prev_acc = "Medium"
        for t in range(steps):
            diff = DIFFS[rng.integers(3)]
            q = 1 / (1 + np.exp(-(np.log(DIFF_ACC[diff] / (1 - DIFF_ACC[diff])) + ability)))
            correct = bool(rng.random() < q)
            if s == 0:      # over-confident
                pc = [0.10, 0.30, 0.60]
            elif s == 2:    # under-confident
                pc = [0.60, 0.30, 0.10]
            else:           # well-calibrated: confidence tracks q
                pc = [max(0.05, 1 - q - 0.15), 0.30, max(0.05, q - 0.15)]
                pc = list(np.array(pc) / np.sum(pc))
            conf = LEVELS[rng.choice(3, p=pc)]
            hint = "Yes" if rng.random() < {"Easy": .1, "Medium": .3, "Hard": .5}[diff] else "No"
            tm = "Fast" if rng.random() < 0.5 else "Slow"
            p_bn = lookup[(conf, diff, tm, hint, prev_acc)]
            rows.append(dict(group=g, step=t, true_state=s, confidence=conf, correct=int(correct),
                             p_bn=p_bn, difficulty=diff))
            prev_acc = "High" if correct else "Low"
            s = rng.choice(3, p=A[s])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Real log loader (sessions = runs separated by > 30 minutes)
# ----------------------------------------------------------------------------

def load_real(path):
    df = pd.read_csv(path)
    conf_col = "effective_confidence" if "effective_confidence" in df.columns else "confidence"
    # Effective confidence may be numeric-blended; the HMM wants Low/Medium/High
    if pd.api.types.is_numeric_dtype(df[conf_col]):
        v = df[conf_col].astype(float)
        df["confidence"] = np.where(v < 0.34, "Low", np.where(v < 0.67, "Medium", "High"))
    else:
        df["confidence"] = df[conf_col]
    df["correct"] = df["correct"].astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "false": 0, "0": 0})
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["confidence", "correct", "p_correct", "ts"]).sort_values("ts")
    df["p_bn"] = df["p_correct"].astype(float)
    gap = df["ts"].diff().dt.total_seconds().fillna(1e9) > 30 * 60
    df["group"] = gap.cumsum()
    df["step"] = df.groupby("group").cumcount()
    df["true_state"] = -1
    return df[["group", "step", "true_state", "confidence", "correct", "p_bn"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Causal (past-only) features
# ----------------------------------------------------------------------------

def build_features(df):
    hmm_model = build_hmm()
    A, B, pi = hmm_model.transmat_, hmm_model.emissionprob_, hmm_model.startprob_
    feats = {k: [] for k in ["hmm0", "hmm1", "hmm2", "rule0", "rule1", "rule2",
                             "roll_acc", "conf", "idx"]}
    conf_num = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
    out_rows = []
    for g, d in df.groupby("group", sort=False):
        alpha = None
        hist = []
        for idx, r in zip(d.index, d.itertuples()):
            # Belief about the state at THIS step, using only earlier answers
            prior = pi if alpha is None else alpha @ A
            rule = rolling_calibration_state(hist) if hist else "Well-calibrated"
            roll = np.mean([c for _, c in hist[-5:]]) if hist else 0.5
            out_rows.append((idx, *prior, *(float(rule == s) for s in STATES),
                             roll, conf_num[r.confidence]))
            # Now reveal this step's answer and update the filter
            o = encode_observation(r.confidence, bool(r.correct))
            a = (prior * B[:, o])
            alpha = a / a.sum()
            hist.append((r.confidence, bool(r.correct)))
    f = pd.DataFrame(out_rows, columns=["idx", "hmm0", "hmm1", "hmm2", "rule0", "rule1", "rule2",
                                        "roll_acc", "conf"]).set_index("idx").sort_index()
    return pd.concat([df, f], axis=1)


def design(df, name):
    z = np.log(np.clip(df.p_bn, 1e-4, 1 - 1e-4) / (1 - np.clip(df.p_bn, 1e-4, 1 - 1e-4)))
    cols = [z]
    c = df.conf - 0.5
    if name == "bn":
        pass
    elif name == "bn+conf":
        cols += [c]
    elif name == "bn+hist":
        cols += [df.roll_acc, df.roll_acc * c, c]
    elif name == "bn+rule":
        cols += [df.rule0, df.rule2, df.rule0 * c, df.rule2 * c, c]
    elif name == "bn+hmm":
        cols += [df.hmm0, df.hmm2, df.hmm0 * c, df.hmm2 * c, c]
    elif name == "bn+oracle":
        cols += [(df.true_state == 0).astype(float), (df.true_state == 2).astype(float),
                 (df.true_state == 0) * c, (df.true_state == 2) * c, c]
    return np.column_stack(cols)


MODELS = ["bn", "bn+conf", "bn+hist", "bn+rule", "bn+hmm"]


# ----------------------------------------------------------------------------
# Grouped repeated CV
# ----------------------------------------------------------------------------

def grouped_cv(df, models):
    groups = df.group.unique()
    y = df.correct.values
    res = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(SEED + r)
        perm = rng.permutation(groups)
        fold_of = {g: i % N_FOLDS for i, g in enumerate(perm)}
        fold = df.group.map(fold_of).values
        for k in range(N_FOLDS):
            tr, te = fold != k, fold == k
            if len(np.unique(y[te])) < 2:
                continue
            for m in models:
                X = design(df, m)
                if m == "bn":
                    p = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                else:
                    sc = StandardScaler().fit(X[tr])
                    p = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[tr]), y[tr]) \
                        .predict_proba(sc.transform(X[te]))[:, 1]
                res.append(dict(repeat=r, fold=k, model=m,
                                brier=float(np.mean((p - y[te]) ** 2)),
                                logloss=float(log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))),
                                auc=float(roc_auc_score(y[te], p))))
            # Raw BN as-is (no recalibration), for reference
            p_raw = df.p_bn.values[te]
            res.append(dict(repeat=r, fold=k, model="bn_raw",
                            brier=float(np.mean((p_raw - y[te]) ** 2)),
                            logloss=float(log_loss(y[te], np.clip(p_raw, 1e-6, 1 - 1e-6))),
                            auc=float(roc_auc_score(y[te], p_raw))))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to real interaction_log.csv")
    ap.add_argument("--learners", type=int, default=200)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("plots", exist_ok=True)

    if args.log:
        df = load_real(args.log)
        source = "REAL LOG"
        models = MODELS
        print(f"Loaded {len(df)} real interactions in {df.group.nunique()} sessions.")
        if df.group.nunique() < N_FOLDS:
            print(f"Need at least {N_FOLDS} sessions for grouped CV; found {df.group.nunique()}. Stopping.")
            return
    else:
        df = simulate(args.learners)
        source = "SYNTHETIC"
        models = MODELS + ["bn+oracle"]
        print(f"Simulated {len(df)} interactions from {df.group.nunique()} learners "
              f"(mechanism check, NOT evidence about real learners).")
    df = build_features(df)

    # ---- state recovery (synthetic only) ------------------------------------
    if source == "SYNTHETIC":
        # filtered belief at step t is about the state at t, given answers before t
        hmm_pred = df[["hmm0", "hmm1", "hmm2"]].values.argmax(1)
        rule_pred = df[["rule0", "rule1", "rule2"]].values.argmax(1)
        base = max(np.mean(df.true_state == s) for s in range(3))
        print(f"\nState recovery (past-only): HMM filter = {np.mean(hmm_pred == df.true_state):.3f}, "
              f"rule-based = {np.mean(rule_pred == df.true_state):.3f}, "
              f"always-majority = {base:.3f}")

    res = grouped_cv(df, models)
    res.to_csv(f"results/hmm_ablation_{source.lower().replace(' ', '_')}.csv", index=False)

    # ---- summary ------------------------------------------------------------
    order = ["bn_raw"] + models
    summ = res.groupby("model")[["brier", "logloss", "auc"]].agg(["mean", "std"]).loc[order]
    print(f"\n=== Grouped-CV results [{source}] -- {N_FOLDS}-fold x {N_REPEATS} repeats, "
          f"split by learner/session ===")
    print(summ.round(4).to_string())

    piv = res.pivot_table(index=["repeat", "fold"], columns="model", values="brier")
    print("\nPaired Brier differences, same folds (negative = first model better):")
    pairs = []
    for a, b in [("bn+hmm", "bn"), ("bn+hmm", "bn+conf"), ("bn+hmm", "bn+hist"),
                 ("bn+hmm", "bn+rule"), ("bn+rule", "bn+conf")] + \
                ([("bn+oracle", "bn+hmm")] if "bn+oracle" in models else []):
        d = (piv[a] - piv[b]).dropna()
        # fold results within a repeat are correlated across repeats; use the
        # per-repeat mean as the unit for the standard error
        per_rep = d.groupby(level=0).mean()
        pairs.append(dict(A=a, B=b, mean_diff=d.mean(),
                          se_over_repeats=per_rep.std(ddof=1) / np.sqrt(len(per_rep)),
                          folds_A_better=int((d < 0).sum()), n_folds=len(d)))
    pairs = pd.DataFrame(pairs)
    print(pairs.round(5).to_string(index=False))
    pairs.to_csv(f"results/hmm_ablation_pairs_{source.lower().replace(' ', '_')}.csv", index=False)

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = {"bn_raw": "#bbbbbb", "bn": "#7a7a7a", "bn+conf": "#4f81bd", "bn+hist": "#e8a33d",
            "bn+rule": "#8064a2", "bn+hmm": "#2e8b57", "bn+oracle": "#222222"}
    for a, met, title in [(ax[0], "brier", "Brier (lower better)"), (ax[1], "auc", "AUC")]:
        m = [x for x in order]
        mu = [summ.loc[x, (met, "mean")] for x in m]
        sd = [summ.loc[x, (met, "std")] for x in m]
        a.bar(range(len(m)), mu, yerr=sd, capsize=3, color=[cols[x] for x in m])
        a.set_xticks(range(len(m))); a.set_xticklabels(m, rotation=30, ha="right")
        lo, hi = min(mu) - 3 * max(sd), max(mu) + 3 * max(sd)
        a.set_ylim(lo, hi); a.set_title(title); a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"HMM ablation: predicting next-answer correctness [{source}], grouped CV; "
                 f"y-axis truncated", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"plots/hmm_ablation_{source.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"\nSaved results/hmm_ablation_*.csv and plots/hmm_ablation_{source.lower().replace(' ', '_')}.png")


if __name__ == "__main__":
    main()
