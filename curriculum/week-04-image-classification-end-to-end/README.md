# Week 4 — Image classification end to end

> **Goal:** by Sunday you can take a real image dataset from folders to a trained, honestly-evaluated classifier and defend every decision from first principles: (1) build a leakage-free data pipeline and argue, in the language of statistical estimation, why each split boundary and sampling choice is where the experiment is won or lost; (2) evaluate with proper scoring rules — not just accuracy — report per-class precision/recall with confidence intervals, read a confusion matrix, and assess whether the model's probabilities are *calibrated*; (3) derive the training-dynamics you rely on — the SGD step-size ceiling, why momentum and AdamW behave as they do, what batch normalization actually normalizes, and why data augmentation is a generalization tool — and diagnose any stalled run from its loss curves; and (4) handle class imbalance and long tails with weighted losses, resampling, and focal loss, choosing the remedy by the real cost of each error.

Week 3 trained a CNN on a clean benchmark where the data pipeline was handed to you and 'accuracy' was a fair summary. Real classification is none of those things. Images arrive as folders of uneven size and quality; classes are imbalanced and often long-tailed; near-duplicate photos of the same subject cluster together and quietly leak across splits; and a single accuracy number hides which classes fail, whether the model's confidence means anything, and how much the estimate would move on a fresh test set. This week you build the **full pipeline** the way a research group or a production team would — and, at graduate depth, you learn *why* each piece is there.

The framing that makes this rigorous is **statistical estimation**. Your test accuracy is not a fact about your model; it is a *point estimate of the population risk* computed from a finite, possibly biased, sample. Everything that goes wrong in real classification — leakage, imbalance, miscalibration, distribution shift — is a way that estimate becomes untrustworthy. So we treat splitting as experimental design, metrics as estimators with error bars, augmentation and weight decay as capacity control that trades a little bias for a lot of variance, and the optimizer and learning-rate schedule as the machinery that actually reaches a good solution on a non-convex landscape. You will meet proper scoring rules and reliability diagrams (Guo et al., 2017), the convergence and implicit-bias story of SGD, batch normalization's real mechanism (Santurkar et al., 2018), and focal loss for the long tail (Lin et al., 2017). This is the exact template your capstone will follow — but you will now be able to defend it line by line.

## Learning objectives

By the end of this week, you will be able to:

- **Design** a leakage-free `Dataset`/`DataLoader` pipeline with group-aware train/val/test splits, and justify every split boundary as an experiment-design decision that keeps the test estimate unbiased.
- **Evaluate** with proper scoring rules and the right summary metrics — per-class precision/recall, macro vs. micro F1, top-k, confusion matrix — and attach a confidence interval so an accuracy number carries its own uncertainty.
- **Assess** whether a classifier is *calibrated* using a reliability diagram and Expected Calibration Error, and apply temperature scaling to fix over-confidence.
- **Regularize** deliberately with augmentation, weight decay, dropout, label smoothing, and early stopping, explaining each as a bias/variance trade rather than a knob.
- **Derive** the SGD step-size ceiling and explain momentum, AdamW, and batch normalization from their update rules and their effect on the loss landscape.
- **Diagnose** any training run from its loss curves — under/overfitting, divergence, plateau, instability — and confirm bugs with the 'overfit a tiny subset' test.
- **Handle** class imbalance and long-tailed distributions with weighted loss, resampling, and focal loss, selecting the remedy from the real cost of each error type.
- **Report** an end-to-end result honestly: a single-touch test evaluation with per-class metrics, calibration, a named error analysis, and an explicit statement of what the number does and does not establish.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 231N` — build an image classifier end to end, from data pipeline through loss, optimization and regularization, and evaluate it with the appropriate metrics. |
| Industry | Report a classifier's performance to people who will act on it: per-class numbers, an interval around the headline figure, whether the probabilities are calibrated, and the named failure modes. |
| Beyond the bar | makes the learner break five training runs on purpose, one named fault each, and build the symptom-to-cause-to-fix guide from the curves those runs produce — `challenges/challenge-02-diagnose-a-broken-run.md` |

## Prerequisites

- Week 3's mini-project (a CNN trained on a clean benchmark), committed and working.
- PyTorch tensors, `nn.Module`, autograd, and a basic training loop.
- Probability: random variables, expectation, variance, the Bernoulli/Binomial and Gaussian distributions, and the idea of an estimator with a standard error.
- Multivariable calculus and linear algebra at the level of gradients, eigenvalues, and quadratic forms.

## This week

**Lectures**

1. [Lecture 1 — The data pipeline is half the job: splitting as experiment design](lecture-notes/01-the-data-pipeline.md)
2. [Lecture 2 — Metrics that tell the truth: proper scoring, calibration, and the regularization toolbox](lecture-notes/02-metrics-and-regularization.md)
3. [Lecture 3 — Learning rates, schedules & diagnosing training](lecture-notes/03-training-dynamics.md)
4. [Lecture 4 — Why the optimizer works: SGD noise, momentum, AdamW, and what batch norm really does](lecture-notes/04-optimization-and-normalization-theory.md)
5. [Lecture 5 — The long tail: imbalanced learning, focal loss, and reporting under shift](lecture-notes/05-imbalance-longtail-and-honest-reporting.md)

**Exercises**

1. [Exercise 1 — Build a group-aware, leakage-free Dataset](exercises/exercise-01-custom-dataset.md)
2. [Exercise 2 — A full metrics report with error bars and calibration](exercises/exercise-02-metrics-report.md)
3. [Exercise 3 — Close the overfitting gap, with evidence](exercises/exercise-03-regularize-and-schedule.md)
4. [Exercise 4 — Long-tailed training with weighting and focal loss](exercises/exercise-04-imbalance-and-focal.md)

**Challenges**

1. [Challenge 1 — Tame an imbalanced dataset and defend the remedy](challenges/challenge-01-imbalance-in-the-wild.md)
2. [Challenge 2 — Diagnose deliberately broken training](challenges/challenge-02-diagnose-a-broken-run.md)
3. [Challenge 3 — Calibration, abstention, and reporting under shift (open)](challenges/challenge-03-calibration-and-shift-open.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 5.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A complete classification pipeline on a real multi-class dataset (e.g. a subset of Oxford-IIIT Pets, or a folder dataset of your own): a group-aware, leakage-free custom `Dataset` with train/val/test; augmented training with a learning-rate schedule and early stopping; explicit imbalance handling where needed; and a report that touches the test set once and gives accuracy with a confidence interval, per-class precision/recall, a confusion matrix, a reliability diagram with ECE before and after temperature scaling, and an error analysis naming the failure modes.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
