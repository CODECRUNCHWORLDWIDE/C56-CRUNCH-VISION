# Week 4 — Homework

Cement the full pipeline — it is the capstone template — and push it to graduate rigor: honest estimates with error bars, calibrated probabilities, a defensible optimizer/schedule, and explicit handling of the long tail. Do the reasoning in writing before you code; the judgment is the skill a library cannot hide.

## Tasks

- Write a checklist of every place data leakage can sneak into a vision pipeline (duplicates, group structure, preprocessing fit on full data, repeated test evaluation) and how to prevent each; apply it to your own dataset in writing.
- For your dataset, justify in writing which metric you would optimize (accuracy, macro-F1, balanced accuracy, recall) and why, in terms of the real cost of each error type, and attach a confidence interval to your current accuracy.
- Add a reliability diagram and Expected Calibration Error to your mini-project, then temperature-scale on validation and report ECE before and after, confirming accuracy is unchanged.
- Add a cosine LR schedule with warmup to your mini-project and switch Adam+weight_decay to AdamW; report the change in accuracy and calibration.
- Construct a long-tailed version of your data and compare plain cross-entropy against focal loss (γ=2) and class-balanced weighting; report per-class recall and balanced accuracy for each.
- Read Guo et al. (2017) and Santurkar et al. (2018) (in the reading list) and write a short note: three things you will change about how you train and report because of them.

## Definition of done

A committed pipeline on a real multi-class dataset: a group-aware, leakage-audited custom `Dataset` with train/val/test and train-only normalization; a regularized CNN trained with a proper optimizer (AdamW or SGD+momentum), an LR schedule, and early stopping, with logged curves; explicit imbalance handling with per-class recall and balanced accuracy where relevant; and a single-touch test report giving accuracy with a confidence interval, per-class precision/recall, a confusion matrix, top-k where relevant, a reliability diagram with ECE before/after temperature scaling, an error analysis naming failure modes, and an honest limitations section. You can defend every design choice from the week's theory.

Submit by committing your work to your course repo under `week-04/`.
