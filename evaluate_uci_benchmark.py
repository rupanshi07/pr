"""
Track II validation on the UCI Student Performance data.

Adds, on the SAME 50 cross-validation folds as evaluate_uci_bn.py:

1. Baseline benchmark   Naive Bayes BN, standard BN (original 3-parent and flat
                        5-parent), multi-layer DAG, plus logistic regression and an
                        MLP trained on the same one-hot features.
2. Significance tests   Nadeau-Bengio corrected resampled t-test (accounts for the
                        overlapping training sets of CV folds), Holm-adjusted over
                        all model pairs; Wilcoxon signed-rank as a secondary check.
3. Curves               Mean ROC and precision-recall curves with a band across the
                        10 CV repeats (positive class = Fail).
4. Posterior convergence  Exact Dirichlet/Beta posterior of P(Pass | parent
                        configuration) as the training set grows, with 95% credible
                        intervals, comparing the 18-configuration network (orig3 /
                        layered Pass node) with the 72-configuration flat5 network.

Run from the project root (needs data/student_performance_raw.csv):
    python evaluate_uci_benchmark.py            # ~3-4 min
    python evaluate_uci_benchmark.py --quick    # 2 repeats
"""

import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve)

from uci_eval_lib import (load_discretized, fit_structure, predict_p_pass, ece,
                          SCENARIOS, STATE_NAMES)

warnings.filterwarnings("ignore")
SEED = 42
N_SPLITS = 5
MODELS = {
    "full": ["naive_bn", "flat5_bn", "layered_bn", "logreg", "mlp"],
    "early": ["orig3_bn(const)", "naive_bn", "flat5_bn", "layered_bn", "logreg", "mlp"],
}
BN_STRUCT = {"naive_bn": "naive", "flat5_bn": "flat5", "layered_bn": "layered",
             "orig3_bn(const)": "orig3"}
COL = {"naive_bn": "#c0504d", "flat5_bn": "#4f81bd", "layered_bn": "#2e8b57",
       "orig3_bn(const)": "#7a7a7a", "logreg": "#e8a33d", "mlp": "#222222"}


def onehot(df, cols):
    parts = [pd.get_dummies(pd.Categorical(df[c], categories=STATE_NAMES[c]), prefix=c)
             for c in cols]
    return pd.concat(parts, axis=1).astype(float).values


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(
        brier=float(np.mean((p - y) ** 2)),
        logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5,
        pr_auc=float(average_precision_score(y, p)),
        ece=ece(p, y),
    )


# ----------------------------------------------------------------------------
# Cross-validated predictions for every model on identical folds
# ----------------------------------------------------------------------------

def run_cv(df, n_repeats):
    n = len(df)
    y = (df["Pass"] == "No").astype(int).values          # 1 = Fail
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    oof = {(s, m): np.full((n_repeats, n), np.nan) for s in SCENARIOS for m in MODELS[s]}
    fold_rows = []
    t0 = time.time()
    for k, (tr, te) in enumerate(rskf.split(df, df["Pass"])):
        r = k // N_SPLITS
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        y_tr, y_te = y[tr], y[te]
        bns = {s: fit_structure(s, train) for s in ["orig3", "naive", "flat5", "layered"]}
        for scen, observed in SCENARIOS.items():
            Xtr, Xte = onehot(train, observed), onehot(test, observed)
            preds = {}
            for m in MODELS[scen]:
                if m in BN_STRUCT:
                    preds[m] = 1 - predict_p_pass(bns[BN_STRUCT[m]], test, observed)
            preds["logreg"] = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, y_tr).predict_proba(Xte)[:, 1]
            preds["mlp"] = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=2000,
                                         random_state=SEED + k).fit(Xtr, y_tr).predict_proba(Xte)[:, 1]
            for m, p in preds.items():
                oof[(scen, m)][r, te] = p
                fold_rows.append(dict(scenario=scen, model=m, repeat=r, fold=k, **metrics(p, y_te)))
        if (k + 1) % 10 == 0:
            print(f"  fold {k + 1}/{N_SPLITS * n_repeats}  ({time.time() - t0:.0f}s)")
    return y, oof, pd.DataFrame(fold_rows)


# ----------------------------------------------------------------------------
# Significance testing
# ----------------------------------------------------------------------------

def nadeau_bengio(d, n_train, n_test):
    """Corrected resampled t-test. d = per-fold metric differences (J values).
    Variance is inflated by (1/J + n_test/n_train) instead of 1/J to account for
    the dependence between overlapping training sets (Nadeau & Bengio, 2003)."""
    J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = np.mean(d) / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=J - 1))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def significance(folds, n):
    n_test = n // N_SPLITS
    n_train = n - n_test
    rows = []
    for scen in SCENARIOS:
        models = [m for m in MODELS[scen] if m != "orig3_bn(const)"] if scen == "full" else MODELS[scen]
        for metric, lower_better in [("brier", True), ("logloss", True),
                                      ("roc_auc", False), ("pr_auc", False)]:
            piv = folds[(folds.scenario == scen)].pivot_table(
                index="fold", columns="model", values=metric)
            pairs = list(itertools.combinations(models, 2))
            raw, recs = [], []
            for a, b in pairs:
                d = (piv[a] - piv[b]).values
                t, p = nadeau_bengio(d, n_train, n_test)
                try:
                    wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    wp = 1.0
                better = (d < 0).mean() if lower_better else (d > 0).mean()
                recs.append(dict(scenario=scen, metric=metric, A=a, B=b,
                                 mean_diff_A_minus_B=float(d.mean()), t_corrected=t,
                                 p_nb=p, p_wilcoxon_uncorrected=float(wp),
                                 frac_folds_A_better=float(better)))
                raw.append(p)
            adj = holm(np.array(raw))
            for rec, a in zip(recs, adj):
                rec["p_nb_holm"] = float(a)
                rec["significant_0.05"] = bool(a < 0.05)
            rows += recs
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------------

def plot_curves(y, oof, scen, path_roc, path_pr):
    grid = np.linspace(0, 1, 101)
    models = [m for m in MODELS[scen] if m != "orig3_bn(const)"]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    fig2, ax2 = plt.subplots(figsize=(6.2, 6))
    for m in models:
        P = oof[(scen, m)]
        tprs, precs, aucs, aps = [], [], [], []
        for r in range(P.shape[0]):
            fpr, tpr, _ = roc_curve(y, P[r])
            tprs.append(np.interp(grid, fpr, tpr))
            pr, rc, _ = precision_recall_curve(y, P[r])
            # sklearn returns recall in decreasing order; reverse for np.interp
            precs.append(np.interp(grid, rc[::-1], pr[::-1]))
            aucs.append(roc_auc_score(y, P[r])); aps.append(average_precision_score(y, P[r]))
        tprs, precs = np.array(tprs), np.array(precs)
        ax.plot(grid, tprs.mean(0), color=COL[m], lw=1.8,
                label=f"{m}  AUC={np.mean(aucs):.3f}")
        ax.fill_between(grid, tprs.min(0), tprs.max(0), color=COL[m], alpha=0.12)
        ax2.plot(grid, precs.mean(0), color=COL[m], lw=1.8,
                 label=f"{m}  AP={np.mean(aps):.3f}")
        ax2.fill_between(grid, precs.min(0), precs.max(0), color=COL[m], alpha=0.12)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall of Fail)")
    ax.set_title(f"ROC (Fail = positive), {scen} evidence\nout-of-fold, mean and min-max over repeats")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    base = y.mean()
    ax2.axhline(base, ls="--", color="gray", lw=1, label=f"chance = {base:.3f}")
    ax2.set_xlabel("Recall (Fail)"); ax2.set_ylabel("Precision (Fail)")
    ax2.set_title(f"Precision-recall (Fail = positive), {scen} evidence")
    ax2.set_ylim(0, 1); ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path_roc, dpi=150)
    fig2.tight_layout(); fig2.savefig(path_pr, dpi=150)
    plt.close(fig); plt.close(fig2)


def plot_pvalue_heatmaps(sig, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    models = [m for m in MODELS["full"]]
    for ax, metric in zip(axes, ["brier", "pr_auc"]):
        M = np.full((len(models), len(models)), np.nan)
        s = sig[(sig.scenario == "full") & (sig.metric == metric)]
        for _, r in s.iterrows():
            i, j = models.index(r.A), models.index(r.B)
            M[i, j] = M[j, i] = r.p_nb_holm
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn_r")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{metric}: Holm-adjusted corrected-t p-values (full evidence)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="adjusted p (green = significant)")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------------
# Posterior convergence
# ----------------------------------------------------------------------------

def posterior_convergence(df, path_png, path_csv, n_perm=30):
    """For a parent-configuration c, the BDeu-prior posterior of theta_c = P(Pass|c) is
    Beta(a + n_c_yes, a + n_c_no), a = ESS / (2 * #configs). We grow the training set
    from 20 rows to all 649 (random orderings) and track the posterior mean and 95%
    credible interval; flat5 (72 configs) is compared with orig3/layered (18 configs)."""
    ESS = 10
    ys = (df["Pass"] == "Yes").values.astype(int)
    designs = {
        "orig3/layered Pass node (18 configs)": ["StudyTime", "Failures", "Absences"],
        "flat5 Pass node (72 configs)": ["StudyTime", "Failures", "Absences", "Medu", "Higher"],
    }
    sizes = np.unique(np.round(np.geomspace(20, len(df), 14)).astype(int))
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(df)) for _ in range(n_perm)]

    out_rows, width_curves = [], {}
    focus = {  # three illustrative configurations of the 3-parent node
        "High study, no failures, low absences": ("High", "None", "Low"),
        "Medium study, no failures, medium absences": ("Medium", "None", "Medium"),
        "Low study, failures, high absences": ("Low", "One_or_more", "High"),
    }
    for dname, cols in designs.items():
        keys = df[cols].apply(tuple, axis=1).values if len(cols) > 1 else df[cols[0]].values
        keys = np.array([hash(k) for k in keys])
        uniq = {k: i for i, k in enumerate(np.unique(keys))}
        kidx = np.array([uniq[k] for k in keys])
        n_cfg_total = int(np.prod([len(STATE_NAMES[c]) for c in cols]))
        a = ESS / (2 * n_cfg_total)
        full_counts = np.zeros((len(uniq), 2))
        for i, c in zip(kidx, ys):
            full_counts[i, c] += 1
        weights = full_counts.sum(1) / full_counts.sum()          # config frequency
        full_mean = (a + full_counts[:, 1]) / (2 * a + full_counts.sum(1))
        w_curve, e_curve = [], []
        for n in sizes:
            ws, es = [], []
            for p in perms:
                sub = p[:n]
                cnt = np.zeros((len(uniq), 2))
                np.add.at(cnt, (kidx[sub], ys[sub]), 1)
                lo = stats.beta.ppf(0.025, a + cnt[:, 1], a + cnt[:, 0])
                hi = stats.beta.ppf(0.975, a + cnt[:, 1], a + cnt[:, 0])
                mean = (a + cnt[:, 1]) / (2 * a + cnt.sum(1))
                ws.append(np.sum(weights * (hi - lo)))
                es.append(np.sum(weights * np.abs(mean - full_mean)))
            w_curve.append((np.mean(ws), np.std(ws))); e_curve.append((np.mean(es), np.std(es)))
            out_rows.append(dict(design=dname, n_train=int(n), mean_ci_width=np.mean(ws),
                                 mean_abs_dev_from_full=np.mean(es)))
        width_curves[dname] = (np.array(w_curve), np.array(e_curve))

    # focus-configuration traces for the 3-parent node
    cols3 = designs["orig3/layered Pass node (18 configs)"]
    a3 = ESS / (2 * 18)
    traces = {k: [] for k in focus}
    for n in sizes:
        acc = {k: [] for k in focus}
        for p in perms:
            sub = df.iloc[p[:n]]
            for k, cfg in focus.items():
                m = (sub[cols3[0]] == cfg[0]) & (sub[cols3[1]] == cfg[1]) & (sub[cols3[2]] == cfg[2])
                ny, nn = int(((sub["Pass"] == "Yes") & m).sum()), int(((sub["Pass"] == "No") & m).sum())
                acc[k].append((stats.beta.mean(a3 + ny, a3 + nn),
                               stats.beta.ppf(0.025, a3 + ny, a3 + nn),
                               stats.beta.ppf(0.975, a3 + ny, a3 + nn)))
        for k in focus:
            traces[k].append(np.mean(acc[k], axis=0))
    traces = {k: np.array(v) for k, v in traces.items()}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    cols_c = ["#c0504d", "#2e8b57", "#4f81bd"]
    for (k, tr), c in zip(traces.items(), cols_c):
        axes[0].plot(sizes, tr[:, 0], color=c, lw=1.8, label=k)
        axes[0].fill_between(sizes, tr[:, 1], tr[:, 2], color=c, alpha=0.15)
    axes[0].set_xscale("log"); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("training rows"); axes[0].set_ylabel("posterior P(Pass | config)")
    axes[0].set_title("Posterior mean and 95% credible interval\n(averaged over 30 random orderings)", fontsize=10)
    axes[0].legend(fontsize=7, loc="lower right"); axes[0].grid(alpha=0.25)
    for (dname, (wc, ec)), c in zip(width_curves.items(), ["#2e8b57", "#4f81bd"]):
        axes[1].plot(sizes, wc[:, 0], color=c, lw=1.8, label=dname)
        axes[1].fill_between(sizes, wc[:, 0] - wc[:, 1], wc[:, 0] + wc[:, 1], color=c, alpha=0.15)
        axes[2].plot(sizes, ec[:, 0], color=c, lw=1.8, label=dname)
        axes[2].fill_between(sizes, ec[:, 0] - ec[:, 1], ec[:, 0] + ec[:, 1], color=c, alpha=0.15)
    axes[1].set_title("Mean 95% credible-interval width\n(weighted by configuration frequency)", fontsize=10)
    axes[2].set_title("Mean |posterior mean - full-data posterior mean|", fontsize=10)
    for ax in axes[1:]:
        ax.set_xscale("log"); ax.set_xlabel("training rows"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for ax in axes:
        ax.set_xticks([20, 50, 100, 200, len(df)])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Posterior convergence of the learned CPD as data grows (BDeu prior, ESS = 10)")
    plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()
    pd.DataFrame(out_rows).to_csv(path_csv, index=False)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_repeats = 2 if args.quick else args.repeats
    os.makedirs("results", exist_ok=True); os.makedirs("plots", exist_ok=True)

    df = load_discretized()
    n = len(df)
    print(f"{n} students, Fail = {(df.Pass == 'No').sum()}; {N_SPLITS}-fold x {n_repeats} repeats.")
    y, oof, folds = run_cv(df, n_repeats)
    folds.to_csv("results/benchmark_fold_metrics.csv", index=False)

    summ = folds.groupby(["scenario", "model"])[["brier", "logloss", "roc_auc", "pr_auc", "ece"]] \
        .agg(["mean", "std"])
    summ.to_csv("results/benchmark_summary.csv")
    pd.set_option("display.width", 220)
    for scen in SCENARIOS:
        print(f"\n=== Benchmark, {scen} evidence (mean over folds, std in brackets) ===")
        t = pd.DataFrame({
            met: [f"{summ.loc[(scen, m), (met, 'mean')]:.3f} ({summ.loc[(scen, m), (met, 'std')]:.3f})"
                  for m in MODELS[scen]]
            for met in ["brier", "logloss", "roc_auc", "pr_auc", "ece"]}, index=MODELS[scen])
        print(t.to_string())

    sig = significance(folds, n)
    sig.to_csv("results/benchmark_significance.csv", index=False)
    print("\n=== Significance (Nadeau-Bengio corrected t, Holm-adjusted), full evidence ===")
    for metric in ["brier", "pr_auc"]:
        s = sig[(sig.scenario == "full") & (sig.metric == metric)]
        print(f"\n[{metric}]")
        print(s[["A", "B", "mean_diff_A_minus_B", "t_corrected", "p_nb", "p_nb_holm",
                 "p_wilcoxon_uncorrected", "frac_folds_A_better"]].round(4).to_string(index=False))
    print(f"\nPairs significant after Holm (p<0.05): "
          f"{int(sig['significant_0.05'].sum())} of {len(sig)} tests.")

    for scen in SCENARIOS:
        plot_curves(y, oof, scen, f"plots/uci_roc_{scen}.png", f"plots/uci_pr_{scen}.png")
    plot_pvalue_heatmaps(sig, "plots/uci_significance_heatmap.png")
    posterior_convergence(df, "plots/uci_posterior_convergence.png",
                          "results/posterior_convergence.csv")
    print("\nSaved: plots/uci_roc_*.png, uci_pr_*.png, uci_significance_heatmap.png, "
          "uci_posterior_convergence.png and results/benchmark_*.csv")


if __name__ == "__main__":
    main()
