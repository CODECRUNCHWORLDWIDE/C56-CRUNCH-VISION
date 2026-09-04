# Week 12 — Capstone: a computer-vision system

> **Goal:** by Sunday you have designed, built, trained, and deployed one computer-vision system on a problem you chose, and — at graduate rigor — you can (1) frame it as classification, detection, or segmentation with a metric matched to the decision it drives and a documented baseline, (2) prove your win over that baseline is real, not sampling noise, using a confidence interval and a paired significance test rather than a single number, (3) evaluate honestly with a visual error analysis, a calibration check, and a per-subgroup fairness audit, and (4) serve the model behind an inference API with verified preprocessing parity and a drift-monitoring plan, packaged as a reproducible repo with a model card that names its failure modes and its legal and privacy constraints.

This is where twelve weeks converge. No new architecture is taught — the capstone asks you to run the entire pipeline you have practiced on a problem *you* pick, and to defend every choice the way a graduate committee or a hiring panel would press you to. The undergraduate version of this week says 'build a model and beat a baseline.' The graduate version says: **a headline number is a hypothesis, not a result** — you will treat your accuracy, mAP, or mIoU as a random variable, put an interval on it, and test whether your model genuinely beats the baseline or merely got a lucky test split.

Vision carries evaluation traps that most ML capstones never confront. Datasets are skewed by geography, skin tone, camera, and context (Buolamwini & Gebru, 2018, 'Gender Shades', FAccT); near-duplicate frames leak across splits and inflate results; deep classifiers are systematically **overconfident**, so their probabilities cannot be trusted without calibration (Guo et al., 2017, 'On Calibration of Modern Neural Networks', ICML); and models that score 95% on a clean test set collapse under mild distribution shift (Hendrycks & Dietterich, 2019, ImageNet-C, ICLR). This week you will measure all of these, not wave at them.

You will also situate your system in the world it must live in. Vision is the most privacy-laden modality in machine learning: faces are biometric identifiers regulated by the EU GDPR (Art. 9) and Illinois' BIPA, and municipalities from San Francisco to the EU AI Act's 'high-risk' Annex III have restricted or banned classes of vision deployment. A capstone that ignores where its model may not legally or ethically be used is incomplete. The deliverable is not a notebook — it is a computer-vision system you would put in front of an employer, a review board, and a skeptic, and defend on the statistics, the failure modes, and the law.

## Learning objectives

By the end of this week, you will be able to:

- **Frame** a real vision problem as classification, detection, or segmentation, choosing the *simplest task that supports the decision*, and justify the choice against annotation cost and label economics.
- **Establish** a documented, reproducible baseline — trivial, classical, or a zero-shot foundation model — and state the exact metric and operating threshold it is measured at.
- **Quantify** uncertainty on your headline metric with a bootstrap or Wilson confidence interval, and **prove** a real improvement over the baseline with a paired test (McNemar for classification, paired bootstrap otherwise).
- **Diagnose** the model with a visual error gallery, a confusion matrix or PR/mAP breakdown, a reliability diagram and expected calibration error, and a robustness probe under corruption and shift.
- **Audit** per-subgroup performance to surface dataset bias, reporting the worst-group metric and the disparity, and name who the system fails and why.
- **Deploy** the model behind a `/predict` API with verified preprocessing parity, a documented export path, and a concrete data-drift monitoring plan for production.
- **Document** the system in a model card and datasheet covering intended use, data provenance and licensing, per-subgroup metrics, failure modes, and privacy/consent/legal constraints (GDPR/BIPA/EU AI Act as applicable).
- **Defend** the project end to end — task, statistics, fairness, and legality — against the toughest technical and ethical critique a reviewer will raise.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 231N` — carry out a substantial vision project on a problem of the learner's own choosing, evaluate it, and present and defend it. |
| Industry | Serve a model behind an API, document what it is for and who it fails, and defend the result to a reviewer whose job is to break it. |
| Beyond the bar | requires a model card, a datasheet and a drift-monitoring plan before the system counts as finished — `exercises/exercise-04-model-card-datasheet-and-drift-plan.md` |

## Prerequisites

- All eleven prior weeks' mini-projects committed and working — classification through Vision Transformers and edge deployment.
- Comfort with a task-appropriate metric: accuracy/precision/recall/F1 (Week 4), mAP at a stated IoU (Week 6), mIoU/Dice (Week 7).
- Basic probability and statistics: sampling distributions, the binomial, confidence intervals, and the idea of a hypothesis test.
- Ability to export and serve a model behind an HTTP endpoint (Week 11), and to pin an environment for reproducibility.

## This week

**Lectures**

1. [Lecture 1 — Scoping a vision project that finishes](lecture-notes/01-scoping-a-vision-project.md)
2. [Lecture 2 — Honest evaluation & error analysis for vision](lecture-notes/02-honest-vision-evaluation.md)
3. [Lecture 3 — Deploying & communicating a vision system](lecture-notes/03-deploy-and-communicate.md)
4. [Lecture 4 — The statistics of a trustworthy result: intervals, significance, and proper scoring](lecture-notes/04-statistics-of-a-trustworthy-result.md)
5. [Lecture 5 — State of the art & responsible deployment: foundation models, shift, and the law](lecture-notes/05-state-of-the-art-and-responsible-deployment.md)

**Exercises**

1. [Exercise 1 — Proposal, baseline ladder, and legal scoping](exercises/exercise-01-proposal-and-baseline.md)
2. [Exercise 2 — End-to-end skeleton with a preprocessing golden test](exercises/exercise-02-end-to-end-skeleton.md)
3. [Exercise 3 — Iterate, then evaluate with statistics and calibration](exercises/exercise-03-iterate-and-evaluate.md)
4. [Exercise 4 — Model card, datasheet, and drift-monitoring plan](exercises/exercise-04-model-card-datasheet-and-drift-plan.md)

**Challenges**

1. [Challenge 1 — Beat the baseline, provably (with an interval and a test)](challenges/challenge-01-beat-your-baseline.md)
2. [Challenge 2 — Demo and defend to a review panel](challenges/challenge-02-demo-and-defend.md)
3. [Challenge 3 — Robustness and foundation-model audit](challenges/challenge-03-robustness-and-foundation-model-audit.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 12.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A polished, reproducible repository: problem framing (task + metric + baseline), a trained vision model that beats its baseline on held-out images with a *stated confidence interval and a paired significance test*, honest evaluation (task metric, visual error gallery, reliability diagram + ECE, per-subgroup fairness audit, robustness probe, named failure modes), a served `/predict` API with verified preprocessing parity and a drift-monitoring plan, a model card + datasheet addressing privacy/consent/legal constraints, and a README anyone can follow to reproduce your results on their own images. Plus a written two-part defense answering the hardest technical and ethical critique.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
