# Week 4 — Image classification end to end

> **Goal:** by Sunday you can take a real image dataset from folders to a trained, honestly-evaluated classifier — with a proper data pipeline, the right metrics, regularization that closes the overfitting gap, and a training run you can diagnose when it stalls.

Week 3 trained a CNN on a clean benchmark. Real classification is messier: images come in folders of varying size and quality, classes are imbalanced, and 'accuracy' hides as much as it reveals. This week you build the **full pipeline** the way you would for a real project — a proper `Dataset`, the right metrics, learning-rate schedules, regularization, and the diagnostic skill to know *why* a run is failing. This is the template your capstone will follow.

## Learning objectives

By the end of this week, you will be able to:

- **Build** a custom `Dataset`/`DataLoader` from an image-folder dataset with train/val/test splits.
- **Choose** and compute the right metrics — accuracy, per-class precision/recall, top-k, confusion matrix.
- **Regularize** deliberately with augmentation, weight decay, dropout, and early stopping.
- **Schedule** learning rates and read a loss curve to diagnose stalls, overfitting, and instability.
- **Evaluate** honestly on held-out data with an error analysis that names failure modes.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — The data pipeline is half the job](lecture-notes/01-the-data-pipeline.md)
2. [Lecture 2 — Metrics that tell the truth, and regularization](lecture-notes/02-metrics-and-regularization.md)
3. [Lecture 3 — Learning rates, schedules & diagnosing training](lecture-notes/03-training-dynamics.md)

**Exercises**

1. [Exercise 1 — Build a custom Dataset with clean splits](exercises/exercise-01-custom-dataset.md)
2. [Exercise 2 — A full metrics report](exercises/exercise-02-metrics-report.md)
3. [Exercise 3 — Close the overfitting gap](exercises/exercise-03-regularize-and-schedule.md)

**Challenges**

1. [Challenge 1 — Tame an imbalanced dataset](challenges/challenge-01-imbalance-in-the-wild.md)
2. [Challenge 2 — Diagnose deliberately broken training](challenges/challenge-02-diagnose-a-broken-run.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 5.

## Deliverable

A complete classification pipeline on a real multi-class dataset (e.g. a subset of Oxford Pets or a folder dataset): custom Dataset, augmented training, a learning-rate schedule, regularization, and a report with per-class metrics, a confusion matrix, and an error analysis.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
