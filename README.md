# Adaptive Tutoring System - Chat-Style Interface

A conversational adaptive tutoring system that dynamically generates questions based on Subject, Topic, and Question Type, then personalizes feedback using Bayesian Networks, Hidden Markov Models, and learner-feedback-driven Reinforcement Learning.

## Overview

The student talks to the tutor like a chat assistant:

1. "What subject would you like to practice?"
2. "What topic within that subject?"
3. "MCQ, Descriptive, or Coding?"

Gemini (with Groq as an automatic fallback provider) generates a fresh question matching those choices. Before the student answers, the system shows a preliminary correctness estimate using only the question''s difficulty and the student''s history. Once they answer -- optionally including a short justification of their reasoning -- a Bayesian Network, a Hidden Markov Model, and a Reinforcement Learning agent jointly decide how to respond.

## Architecture
Subject / Topic / Question Type (chat input)
|
v
Gemini/Groq: generate_question() --> dynamic question + reference answer
|
v
Preliminary Bayesian Network query (Difficulty + PreviousAccuracy only)
--> shown to student BEFORE they answer, as a partial-evidence estimate
|
v
Student sets Confidence (sidebar), optionally requests a Hint (button),
optionally writes a justification for their answer
|
v
IN PARALLEL: Gemini/Groq grades the answer (MCQ = exact string match,
no LLM call) AND analyzes the justification text for hedging vs.
assertive language --> linguistic confidence
|
v
Self-reported confidence + linguistic confidence are blended into one
effective confidence value
|
v
Full Bayesian Network query (all 5 evidence variables now known:
Confidence, Difficulty, Time, Hints, PreviousAccuracy) --> P(Correct)
|
v
Hidden Markov Model (Viterbi decoding over interaction history)
--> Calibration state: Over-confident / Well-calibrated / Under-confident|
v
Reinforcement Learning Agent (tabular Q-learning)
--> Selected Action: Hint / Explanation / Retry / Reveal
|
v
Gemini/Groq: generate_feedback() --> NL text shown to student
|
v
Student rates: Helpful / Not Helpful
|
v
combined_reward = outcome-based reward + satisfaction bonus
|
v
Q-table updated and persisted; interaction logged to CSV
|
v
If action was Retry and the answer was wrong: same question repeats.
Otherwise: next question.

## The Bayesian Network: Structure

The live network is a genuine multi-layer DAG, not a naive/star structure:
PreviousAccuracy --------> Confidence
Difficulty -------------> Hints
Difficulty, Hints -------> Time
Confidence, Difficulty, Time, Hints, PreviousAccuracy -> Correct

Two ways this network is queried:

- Preliminary query (before the student answers): evidence = Difficulty and PreviousAccuracy only. Confidence, Hints, and Time are genuinely marginalized over, using the upstream edges above -- this is real inference under partial evidence.
- Full query (after the student answers): evidence = all five variables, fully observed. This collapses to a direct CPD lookup -- the richer structure specifically enables the preliminary query, which a naive/star structure could not support at all.

CPD values for the Correct node are still expert-specified (a hand-picked linear scoring formula), not learned from data. A separate, standalone network (dataset_bayesian_network.py) demonstrates real parameter learning from the real UCI Student Performance dataset (Cortez, 2008, 649 students) via pgmpy''s BayesianEstimator -- but that network is not wired into the live app, since no public dataset captures live self-reported confidence or hint-usage signals.

## The Hidden Markov Model

Three hidden states (Over-confident, Well-calibrated, Under-confident), inferred via Viterbi decoding over a sequence of (confidence, correctness) observations. The transition, emission, and initial-state matrices are currently hand-specified.

Two supporting pieces exist:

- hmm_state_definition.py gives a formal, checkable, rule-based definition of the three states (a rolling mean calibration-error threshold), independent of the hand-picked matrices.
- learn_hmm_from_log.py attempts real Baum-Welch parameter learning from data/interaction_log.csv, but explicitly refuses to run unless at least 50 real logged interactions exist with at least 4 of 6 possible observation types represented -- this is intentional, to avoid producing an overfit or meaningless model on insufficient data.

## Reinforcement Learning

Tabular Q-learning over 9 states (P(Correct) bucket x calibration state) and 4 actions (Hint, Explanation, Retry, Reveal). Trained offline on 3,000 simulated episodes (train.py). Reward combines an automatic outcome-based signal with a real learner satisfaction rating (Helpful / Not Helpful).

Evaluated rigorously across 10 independent trials against two baselines (evaluate_rl_robust.py): the trained policy clearly beats a random policy, but does not show a statistically convincing advantage over a simple fixed "always Hint" policy at the current sample size -- this is reported honestly rather than only showing favorable comparisons.

## Terminology Note: RLHF

This project uses learner feedback (Helpful / Not Helpful ratings) as part of the reward signal for a tabular Q-learning tutoring policy. This is reinforcement learning that uses human/learner feedback -- but it is NOT the same as conventional NLP RLHF (the technique used to align large language models), which involves training a separate reward model on human preference data and using it to fine-tune the LLM''s own weights via policy-gradient methods.

In this project, Gemini and Groq are never fine-tuned; their weights are never modified. Only the separate, from-scratch Q-learning agent''s policy is updated using learner feedback. The more precise term for this is "learner-feedback-driven reinforcement learning" or "RLHF-inspired reinforcement learning."

## LLM Integration and Resilience

Google Gemini (gemini-flash-latest) is the primary provider for question generation, free-text grading, and feedback text. Groq (openai/gpt-oss-120b) is an automatic fallback if Gemini fails or times out -- both clients have explicit timeouts (30s / 20s) to prevent multi-minute hangs if a request stalls rather than cleanly failing.

If grading itself fails (e.g., both providers are unreachable), the student is told grading failed and is NOT silently marked wrong -- this distinguishes an LLM/API failure from a genuinely incorrect answer.

## Project Structure
Project/
|-- modules/
| |-- bayesian_network.py # Live BN: structure, full query, preliminary query
| |-- hmm_calibration.py # HMM: hand-specified matrices, Viterbi inference
| |-- rl_agent.py # Tabular Q-learning tutoring agent
| |-- tutor.py # Gemini/Groq: questions, grading, feedback, justification analysis
|-- data/
| |-- q_table.json # Persisted, learned Q-table
| |-- interaction_log.csv # Real logged interactions
| |-- student_performance_raw.csv # Generated by fetch_dataset.py (not committed)
|-- fetch_dataset.py # Downloads UCI Student Performance dataset
|-- dataset_bayesian_network.py # Learns CPDs from real data using pgmpy (standalone demo)
|-- hmm_state_definition.py # Formal, rule-based calibration state definition
|-- learn_hmm_from_log.py # Attempts to learn HMM parameters from real logs (gated)
|-- train.py # Batch-trains the RL agent (simulated, no LLM calls)
|-- evaluate.py # Single-run RL vs. fixed baseline + Q-table heatmap
|-- evaluate_rl_robust.py # 10-trial RL vs. random and fixed baselines
|-- evaluate_calibration.py # Brier score + calibration curve on real logged data
|-- app.py # The chat-style Streamlit application
|-- requirements.txt

## Setup

1. Clone the repository and enter the project folder:
git clone https://github.com/rupanshi07/adaptive-tutoring-chatstyle.git
cd adaptive-tutoring-chatstyle

2. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Get a free Gemini API key from https://aistudio.google.com/app/apikey and a free Groq API key from https://console.groq.com/keys, then set them as environment variables:
setx GEMINI_API_KEY "your-gemini-key-here"
setx GROQ_API_KEY "your-groq-key-here"
   Close and reopen your terminal after running this.

## Running the Project

Run the chat-style demo:
streamlit run app.py

Fetch the sourced dataset:
python fetch_dataset.py

Learn Bayesian Network parameters from real data:
python dataset_bayesian_network.py


Attempt to learn HMM parameters from real logged interactions:

python learn_hmm_from_log.py


Train the RL agent (simulated, 3000 episodes):

python train.py


Run rigorous multi-trial RL evaluation:

python evaluate_rl_robust.py


Check Bayesian Network calibration on real logged data:

python evaluate_calibration.py

## Dataset Citation

Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

## Tech Stack

- pgmpy -- Bayesian Network construction, exact inference (Variable Elimination), and parameter learning
- hmmlearn -- Hidden Markov Model, Viterbi decoding, (gated) Baum-Welch learning
- Google Gemini API (google-genai) and Groq API (groq) -- dynamic question generation, grading, feedback, with automatic fallback
- ucimlrepo -- fetching the sourced UCI dataset
- Streamlit -- conversational chat-style web UI

## Known Limitations

- The live Bayesian Network Correct-node CPD is still expert-specified, not learned from real data; a script to relearn it from interaction_log.csv (analogous to learn_hmm_from_log.py) does not exist yet.
- The HMM matrices remain hand-specified; real learning is implemented but correctly gated behind a data-sufficiency check that is not yet met.
- Only a small number of real interactions have been logged so far -- the Brier score/calibration curve and any future data-driven learning are not yet based on a meaningful sample size.
- train.py simulated learner behavior uses an arbitrary difficulty-success heuristic, not grounded in the real UCI dataset actual pass-rate patterns.
- The RL agent shows a clear advantage over random behavior but not a statistically convincing advantage over a simple fixed always-Hint baseline, based on rigorous 10-trial testing.
- The dataset-driven Bayesian Network (UCI-based) has not itself been evaluated for calibration, and the source dataset class imbalance (about 85 percent Pass, 15 percent Fail) was not addressed.

## A Real Result: Learned vs. Hand-Specified Probabilities

Once enough real, varied interaction data accumulated (75 real logged interactions), learn_live_bn_from_log.py successfully learned real CPD parameters for the live Bayesian Network -- the first time this project moved from "the infrastructure exists" to "this actually happened with real data."

For a Medium-confidence, Medium-difficulty, Fast, no-hint, High-previous-accuracy student (a combination observed 26 times in real logged data):

- Hand-specified formula (expert judgment): P(Correct) = 0.700
- Learned from 26 real observations: P(Correct) = 0.996

This is a genuine, measured divergence between expert intuition and real behavior -- not an error to explain away, but the exact justification for why data-driven relearning matters. For combinations with little or no real support in the log, the learned model correctly falls back toward an uninformative estimate near 0.5, rather than expressing false confidence -- demonstrating that Bayesian parameter estimation behaves correctly under both sufficient and insufficient evidence.

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
