## Evaluation of the dataset-driven (UCI) Bayesian Network

All numbers: UCI Student Performance (Portuguese course, 649 students, 100 Fail = 15.4%),
repeated stratified 5-fold CV x 10 (50 folds, ~20 Fail cases per test fold). Positive class = Fail.
G1/G2 are excluded as predictors (G3 is almost a function of them). Reproduce with:

    python fetch_dataset.py
    python evaluate_uci_bn.py            # ~90 s, writes results/*.csv and plots/uci_*.png
    python evaluate_hmm_ablation.py      # synthetic mechanism check
    python evaluate_hmm_ablation.py --log data/interaction_log.csv   # on your real log

### Structures compared
- orig3: StudyTime, Failures, Absences -> Pass (the original network)
- naive: Pass -> each of StudyTime, Failures, Absences, Medu, Higher
- flat5: all five features -> Pass (72-column CPT)
- layered: Medu -> Higher -> {StudyTime, Failures, Absences} -> Pass (18-column Pass CPT)

Two evidence settings: **full** (all five observed) and **early** (only Medu, Higher observed).

### Results (mean over 10 repeats)
| scenario | structure | Brier | ECE | ROC-AUC | PR-AUC (Fail) |
|---|---|---|---|---|---|
| full | orig3 / layered | 0.113 | 0.029 | 0.717 | 0.367 |
| full | naive | 0.115 | 0.055 | 0.779 | 0.408 |
| full | flat5 | 0.117 | 0.067 | 0.697 | 0.361 |
| early | naive | 0.120 | 0.021 | 0.650 | 0.275 |
| early | flat5 | 0.120 | 0.038 | 0.656 | 0.305 |
| early | layered | 0.122 | 0.043 | 0.637 | 0.285 |
| early | prior only | 0.130 | - | 0.5 | 0.154 |

- With all evidence observed, layered and orig3 give *identical* predictions (upstream nodes are
  d-separated from Pass), so layering can only matter under partial evidence.
- The multi-layer DAG did **not** beat the naive BN. Full evidence: paired Brier difference
  layered - naive = -0.0019 +/- 0.0016 (SE), better in 30/50 folds (not significant);
  vs flat5 -0.0040 +/- 0.0013 (33/50). Early evidence: layered is slightly *worse* than naive
  (+0.0022 +/- 0.0008, better in only 16/50 folds). Naive has the best ranking (ROC/PR-AUC)
  but worse calibration (ECE 0.055 vs 0.029).
- flat5 pays for its large CPT (sparse cells), which is the main argument for layering.

### Calibration
Bootstrap 95% CI for ECE of the layered/full model: [0.027, 0.056]; Brier CI [0.095, 0.130].
Cross-fitted Platt recalibration (fitted inside each training fold) has slope < 1 for the
full-evidence models (0.75 layered/orig3, 0.64 naive, 0.44 flat5), i.e. the raw probabilities are
over-confident. It helps naive (Brier 0.1150 -> 0.1097) and flat5 (log-loss 0.420 -> 0.390) but
not layered/orig3 under full evidence. Reliability diagrams: plots/uci_reliability.png.

### Class imbalance
Resampling is applied to training folds only.
| variant | ECE | Fail recall | Fail precision | balanced acc. |
|---|---|---|---|---|
| no resampling, threshold 0.5 | 0.029 | 0.30 | 0.56 | 0.629 |
| no resampling, threshold = training Fail rate | 0.029 | 0.55 | 0.35 | 0.680 |
| oversample, raw probabilities | 0.264 | 0.53 | 0.34 | 0.672 |
| oversample + prior-shift correction | 0.034 | 0.53 | 0.34 | 0.672 |
| undersample, raw probabilities | 0.271 | 0.57 | 0.32 | 0.674 |
| undersample + prior-shift correction | 0.044 | 0.57 | 0.32 | 0.674 |

Resampling without correcting the prior destroys calibration (ECE ~0.27). With correction it
recovers, but it gives no better decisions than simply moving the threshold on the un-resampled
model, which is the cheapest fix.

### HMM ablation (bn vs bn+HMM)
Evaluated as: does the HMM state (forward-filtered, past-only) add information to the BN for
predicting the next answer? Controls: confidence only, rolling accuracy, rule-based state.
On simulated learners (mechanism check only): HMM state recovery 0.42 vs 0.41 majority baseline;
bn+hmm improves Brier over bn by 0.0011, but the *oracle* true state adds only 0.00002 more, and a
plain rolling-accuracy feature beats the HMM (0.0019). So predicting correctness is not where this
HMM earns its keep; the real-log run is the one that counts.

### Limitations of this evaluation
- Features Medu and Higher were added to give the structures something to differ on; the layered
  DAG is expert-specified, not structure-learned.
- ~20 Fail cases per test fold: metric differences of ~0.002 Brier are within noise.
- The HMM numbers above are from a simulator I wrote, not real learners.
