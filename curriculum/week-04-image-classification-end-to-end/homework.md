# Week 4 — Homework

Cement the full pipeline; it is the capstone template.

## Tasks

- Write a checklist of every place data leakage can sneak into a vision pipeline and how to prevent each.
- For your dataset, justify in writing which metric you'd optimize and why, in terms of the cost of each error type.
- Add a cosine LR schedule with warmup to your mini-project and report the accuracy change.
- Read the CS231n notes on training/regularization (in resources) and list three diagnostic tips you'll adopt.

## Definition of done

A committed pipeline on a real multi-class dataset: a leakage-free custom Dataset with train/val/test, augmented training, a regularized CNN trained with an LR schedule, and a report with per-class precision/recall, a confusion matrix, top-k where relevant, an error analysis naming failure modes, and the final held-out accuracy.

Submit by committing your work to your course repo under `week-04/`.
