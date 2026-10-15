"""
Proper evaluation of the dataset-driven (UCI) Bayesian Network.

Covers four things that were missing:

1. Cross-validation     -- repeated stratified 5-fold CV (default 5 x 10 = 50 folds),
                           stratified on Pass so every fold keeps the ~85/15 split.
2. Structure ablation   -- naive Bayes vs flat converging DAG vs multi-layer DAG
                           (and the original 3-parent network), under two evidence
                           scenarios: "full" (everything observed) and "early"
                           (only background variables observed, behaviour not yet).
3. Calibration testing  -- Brier, log-loss, ECE, reliability diagrams, bootstrap CIs,
                           plus cross-fitted Platt recalibration.
4. Class imbalance      -- none / random oversampling / random undersampling (training
                           folds only), with and without Bayes prior-shift correction,
                           reported with imbalance-aware metrics for the Fail class.

Run from the project root:
    python evaluate_uci_bn.py              # full run (50 folds)
    python evaluate_uci_bn.py --quick      # 2 repeats, no recalibration (smoke test)

Needs data/student_performance_raw.csv (python fetch_dataset.py).
Writes CSVs to results/ and figures to plots/.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from uci_eval_lib import (
    load_discretized, STRUCTURES, SCENARIOS, fit_structure, predict_p_pass,
    resample_train, prior_correct, probability_metrics, decision_metrics,
    calibration_bins, ece, fit_platt, apply_platt,
)

warnings.filterwarnings("ignore")
SEED = 42
STRUCT_ORDER = ["orig3", "naive", "flat5", "layered"]
STRATEGIES = ["none", "oversample", "undersample"]
# (variant name, strategy, which probabilities, which threshold)
VARIANTS = [
    ("none_thr0.5", "none", "raw", "half"),
    ("none_thr_prior", "none", "raw", "prior"),
    ("over_raw_thr0.5", "oversample", "raw", "half"),
    ("over_corrected_thr_prior", "oversample", "corrected", "prior"),
    ("under_raw_thr0.5", "undersample", "raw", "half"),
    ("under_corrected_thr_prior", "undersample", "corrected", "prior"),
]


# ----------------------------------------------------------------------------
# Core CV loop
# ----------------------------------------------------------------------------

def run_cv(df, n_repeats, do_recal):
    n = len(df)
    y_fail = (df["Pass"] == "No").astype(int).values
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rng = np.random.default_rng(SEED)

    # preds[(scenario, struct, strategy)] -> dict(raw/corrected/thr arrays [R, N])
    def blank():
        return np.full((n_repeats, n), np.nan)

    store = {}
    platt = {}   # (scenario, struct) -> [R, N] recalibrated p_fail (strategy none)
    prior_only = blank()
    thr_prior = blank()  # per-student threshold = training Fail prevalence of its fold
    fold_brier = {}      # (scenario, struct) -> list of per-fold Brier (none, raw)
    platt_params = {}

    t0 = time.time()
    for k, (tr, te) in enumerate(rskf.split(df, df["Pass"])):
        r = k // 5
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        pi_fail = float((train["Pass"] == "No").mean())
        prior_only[r, te] = pi_fail
        thr_prior[r, te] = pi_fail

        for struct in STRUCT_ORDER:
            for strategy in STRATEGIES:
                tr_rs = resample_train(train, strategy, rng)
                model = fit_structure(struct, tr_rs)
                sample_fail = float((tr_rs["Pass"] == "No").mean())
                for scen, observed in SCENARIOS.items():
                    p_fail = 1 - predict_p_pass(model, test, observed)
                    key = (scen, struct, strategy)
                    if key not in store:
                        store[key] = {"raw": blank(), "corrected": blank()}
                    store[key]["raw"][r, te] = p_fail
                    store[key]["corrected"][r, te] = (
                        p_fail if strategy == "none"
                        else prior_correct(p_fail, pi_fail, sample_fail))
                    if strategy == "none":
                        fb = fold_brier.setdefault((scen, struct), [])
                        fb.append(float(np.mean((p_fail - y_fail[te]) ** 2)))

            if do_recal:
                # Cross-fitted Platt: inner OOF predictions on the training fold only
                inner = StratifiedKFold(5, shuffle=True, random_state=SEED + k)
                for scen, observed in SCENARIOS.items():
                    inner_p = np.full(len(train), np.nan)
                    for itr, ite in inner.split(train, train["Pass"]):
                        m_in = fit_structure(struct, train.iloc[itr].reset_index(drop=True))
                        inner_p[ite] = 1 - predict_p_pass(
                            m_in, train.iloc[ite].reset_index(drop=True), observed)
                    y_tr = (train["Pass"] == "No").astype(int).values
                    a, b = fit_platt(inner_p, y_tr)
                    platt_params.setdefault((scen, struct), []).append((a, b))
                    m_full = fit_structure(struct, train)
                    p_te = 1 - predict_p_pass(m_full, test, observed)
                    platt.setdefault((scen, struct), blank())[r, te] = apply_platt(p_te, a, b)

        if (k + 1) % 10 == 0:
            print(f"  fold {k + 1}/{5 * n_repeats}  ({time.time() - t0:.0f}s)")

    return y_fail, store, platt, platt_params, prior_only, thr_prior, fold_brier


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate(y_fail, store, prior_only, thr_prior, n_repeats):
    rows = []
    for (scen, struct, strategy), d in store.items():
        for vname, vstrat, which, thr in VARIANTS:
            if vstrat != strategy:
                continue
            p_all = d[which]
            per_repeat = []
            for r in range(n_repeats):
                m = probability_metrics(p_all[r], y_fail)
                # Decisions: threshold per student (0.5 or its fold's training prior)
                pred = (p_all[r] >= (0.5 if thr == "half" else thr_prior[r])).astype(int)
                m.update(_decision_from_pred(pred, y_fail))
                per_repeat.append(m)
            rows.append(_summarize(per_repeat, scen, struct, vname))
    # Prior-only reference (always predicts the training Fail prevalence)
    per_repeat = []
    for r in range(n_repeats):
        m = probability_metrics(prior_only[r], y_fail)
        pred = (prior_only[r] >= thr_prior[r]).astype(int)
        m.update(_decision_from_pred(pred, y_fail))
        per_repeat.append(m)
    for scen in SCENARIOS:
        rows.append(_summarize(per_repeat, scen, "prior_only", "none_thr_prior"))
    res = pd.DataFrame(rows)
    # Brier skill score vs prior-only
    ref = res[res.structure == "prior_only"].set_index("scenario")["brier_mean"].to_dict()
    res["brier_skill"] = res.apply(lambda r: 1 - r["brier_mean"] / ref[r["scenario"]], axis=1)
    return res


def _decision_from_pred(pred, y):
    from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)
    return {
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "recall_fail": float(recall_score(y, pred, zero_division=0)),
        "precision_fail": float(precision_score(y, pred, zero_division=0)),
        "f1_fail": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "pred_fail_rate": float(pred.mean()),
    }


def _summarize(per_repeat, scen, struct, variant):
    df = pd.DataFrame(per_repeat)
    row = {"scenario": scen, "structure": struct, "variant": variant}
    for c in df.columns:
        row[f"{c}_mean"] = df[c].mean()
        row[f"{c}_std"] = df[c].std(ddof=1) if len(df) > 1 else 0.0
    return row


def bootstrap_ci(y_fail, P, n_boot=300, seed=SEED):
    """Student-level bootstrap of Brier and ECE for stacked repeat predictions P [R, N]."""
    rng = np.random.default_rng(seed)
    n = len(y_fail)
    b, e = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y_fail[idx]
        b.append(np.mean([np.mean((P[r, idx] - yy) ** 2) for r in range(P.shape[0])]))
        e.append(np.mean([ece(P[r, idx], yy) for r in range(P.shape[0])]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(b), q(e)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

COL = {"orig3": "#7a7a7a", "naive": "#c0504d", "flat5": "#4f81bd", "layered": "#2e8b57"}


def plot_reliability(y_fail, store, platt, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
    for i, scen in enumerate(SCENARIOS):
        for j, struct in enumerate(STRUCT_ORDER):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            p_raw = store[(scen, struct, "none")]["raw"].ravel()
            y_rep = np.tile(y_fail, store[(scen, struct, "none")]["raw"].shape[0])
            mp, fp, cnt = calibration_bins(p_raw, y_rep, 10)
            ok = cnt > 0
            ax.scatter(mp[ok], fp[ok], s=np.sqrt(cnt[ok]) * 6, color=COL[struct],
                       alpha=0.8, label="raw (OOF)", zorder=3)
            ax.plot(mp[ok], fp[ok], color=COL[struct], lw=1)
            if (scen, struct) in platt:
                p_pl = platt[(scen, struct)].ravel()
                mp2, fp2, cnt2 = calibration_bins(p_pl, y_rep, 10)
                ok2 = cnt2 > 0
                ax.plot(mp2[ok2], fp2[ok2], "o-", color="black", ms=3, lw=1,
                        label="Platt (cross-fitted)")
            ax.set_title(f"{struct} / {scen}  (ECE={ece(p_raw, y_rep):.3f})", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("Predicted P(Fail)")
            if j == 0:
                ax.set_ylabel("Observed Fail rate")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Reliability diagrams, out-of-fold predictions (marker area ~ sqrt(count))")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_structure_ablation(res, path):
    metrics = [("brier", "Brier (lower better)"), ("roc_auc", "ROC-AUC"),
               ("pr_auc", "PR-AUC (Fail class)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, scen in enumerate(SCENARIOS):
        for j, (m, label) in enumerate(metrics):
            ax = axes[i, j]
            sub = res[(res.scenario == scen) & (res.variant == "none_thr0.5")
                      & (res.structure.isin(STRUCT_ORDER))].set_index("structure").loc[STRUCT_ORDER]
            ax.bar(STRUCT_ORDER, sub[f"{m}_mean"], yerr=sub[f"{m}_std"], capsize=4,
                   color=[COL[s] for s in STRUCT_ORDER], alpha=0.9)
            lo = sub[f"{m}_mean"].min() - 4 * sub[f"{m}_std"].max()
            hi = sub[f"{m}_mean"].max() + 4 * sub[f"{m}_std"].max()
            ax.set_ylim(max(lo, 0), min(hi, 1))
            ax.set_title(f"{label} -- {scen} evidence", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Structure ablation: mean +/- std across CV repeats (axis truncated)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_imbalance(res, struct, scen, path):
    sub = res[(res.scenario == scen) & (res.structure == struct)].set_index("variant")
    order = [v[0] for v in VARIANTS]
    sub = sub.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(range(len(order)), sub["ece_mean"], yerr=sub["ece_std"], capsize=3, color="#4f81bd")
    axes[0].set_title("ECE of P(Fail) (lower better)")
    x = np.arange(len(order)); w = 0.27
    axes[1].bar(x - w, sub["recall_fail_mean"], w, label="recall (Fail)", color="#2e8b57")
    axes[1].bar(x, sub["precision_fail_mean"], w, label="precision (Fail)", color="#c0504d")
    axes[1].bar(x + w, sub["bal_acc_mean"], w, label="balanced acc.", color="#7a7a7a")
    axes[1].legend(fontsize=8); axes[1].set_title("Decision quality on the minority class")
    axes[2].bar(range(len(order)), sub["mcc_mean"], yerr=sub["mcc_std"], capsize=3, color="#8064a2")
    axes[2].set_title("Matthews corr. coef.")
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Imbalance handling -- {struct}, {scen} evidence")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_repeats = 2 if args.quick else args.repeats
    do_recal = not args.quick

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    df = load_discretized()
    n_fail = int((df["Pass"] == "No").sum())
    print(f"{len(df)} students; Fail = {n_fail} ({n_fail / len(df):.1%}), "
          f"Pass = {len(df) - n_fail} ({1 - n_fail / len(df):.1%})")
    print(f"Repeated stratified 5-fold CV x {n_repeats} = {5 * n_repeats} folds, "
          f"~{n_fail // 5} Fail cases per test fold.\n")

    y_fail, store, platt, platt_params, prior_only, thr_prior, fold_brier = run_cv(
        df, n_repeats, do_recal)
    res = aggregate(y_fail, store, prior_only, thr_prior, n_repeats)

    # --- CSVs --------------------------------------------------------------
    res.to_csv("results/uci_cv_results.csv", index=False)

    # Paired fold-level Brier comparison (same folds for every model)
    pair_rows = []
    for scen in SCENARIOS:
        for a, b in [("layered", "naive"), ("layered", "flat5"), ("layered", "orig3"),
                     ("flat5", "naive")]:
            da = np.array(fold_brier[(scen, a)]); db = np.array(fold_brier[(scen, b)])
            diff = da - db
            pair_rows.append({
                "scenario": scen, "A": a, "B": b, "mean_brier_A_minus_B": diff.mean(),
                "std_error": diff.std(ddof=1) / np.sqrt(len(diff)),
                "folds_A_better": int((diff < 0).sum()), "n_folds": len(diff),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv("results/uci_paired_brier.csv", index=False)

    # Bootstrap CIs for the headline (strategy none, raw) probabilities
    ci_rows = []
    for scen in SCENARIOS:
        for struct in STRUCT_ORDER:
            (bl, bh), (el, eh) = bootstrap_ci(y_fail, store[(scen, struct, "none")]["raw"])
            ci_rows.append({"scenario": scen, "structure": struct,
                            "brier_ci_low": bl, "brier_ci_high": bh,
                            "ece_ci_low": el, "ece_ci_high": eh})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv("results/uci_bootstrap_ci.csv", index=False)

    # Platt recalibration summary
    if do_recal:
        pl_rows = []
        for (scen, struct), P in platt.items():
            raw = store[(scen, struct, "none")]["raw"]
            for tag, arr in [("raw", raw), ("platt", P)]:
                ms = [probability_metrics(arr[r], y_fail) for r in range(n_repeats)]
                pl_rows.append({
                    "scenario": scen, "structure": struct, "probabilities": tag,
                    "brier": np.mean([m["brier"] for m in ms]),
                    "logloss": np.mean([m["logloss"] for m in ms]),
                    "ece": np.mean([m["ece"] for m in ms]),
                    "mean_platt_slope": np.mean([a for a, _ in platt_params[(scen, struct)]]),
                    "mean_platt_intercept": np.mean([b for _, b in platt_params[(scen, struct)]]),
                })
        pd.DataFrame(pl_rows).to_csv("results/uci_platt_recalibration.csv", index=False)

    # --- Plots -------------------------------------------------------------
    plot_reliability(y_fail, store, platt, "plots/uci_reliability.png")
    plot_structure_ablation(res, "plots/uci_structure_ablation.png")
    plot_imbalance(res, "layered", "full", "plots/uci_imbalance_layered_full.png")

    # --- Console summary ---------------------------------------------------
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    fmt = lambda m, s: res[m + "_mean"].round(3).astype(str) + "+-" + res[m + "_std"].round(3).astype(str)
    res["Brier"] = fmt("brier", "brier"); res["ECE"] = fmt("ece", "ece")
    res["ROC"] = fmt("roc_auc", "roc_auc"); res["PR"] = fmt("pr_auc", "pr_auc")
    print("\n=== Structure ablation (no resampling, probabilities as-is) ===")
    print(res[res.variant == "none_thr0.5"][
        ["scenario", "structure", "Brier", "ECE", "ROC", "PR", "brier_skill"]
    ].round(3).to_string(index=False))
    print("\n=== Paired Brier differences (negative = A better), same folds ===")
    print(pairs.round(4).to_string(index=False))
    print("\n=== Bootstrap 95% CIs (student-level) ===")
    print(ci.round(4).to_string(index=False))
    print("\n=== Imbalance handling, layered DAG, full evidence ===")
    sub = res[(res.structure == "layered") & (res.scenario == "full")]
    print(sub[["variant", "Brier", "ECE", "recall_fail_mean", "precision_fail_mean",
               "bal_acc_mean", "mcc_mean", "pred_fail_rate_mean"]].round(3).to_string(index=False))
    print("\nSaved CSVs to results/ and figures to plots/.")


if __name__ == "__main__":
    main()
