# Exercise 3 — Iterate, then evaluate with statistics and calibration

**Goal:** improve the model, then evaluate like a statistician.

## Tasks

1. Improve the model deliberately — transfer learning, the Week-4 training toolbox, augmentation, the right
   architecture — logging what each change bought on *validation* (never the test set).
2. When done improving, run the **full honest evaluation on the untouched test set, once**: the task metric
   vs. baseline *with a confidence interval* (Wilson for accuracy; group-aware bootstrap for mAP/mIoU/F1);
   a paired significance test of the improvement (McNemar for classification, paired bootstrap otherwise);
   a confusion matrix / PR / per-class breakdown; a **reliability diagram + ECE** (apply temperature scaling
   if miscalibrated); and a *visual* error gallery.
3. Run a per-subgroup/bias audit where relevant, reporting the worst-group metric and the disparity, plus a
   robustness probe under a fixed corruption suite.
4. Write the limitations and named failure modes you found.

## Deliverable

An evaluation report with: results-vs-baseline *with intervals and a p-value/significance statement*, a
reliability diagram + ECE, a visual error gallery, a subgroup/bias check, a corruption robustness table, and
a named-limitations section. This report is the heart of the capstone.
