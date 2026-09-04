# Exercise 2 — A full metrics report with error bars and calibration

**Goal:** evaluate like a professional — not one number, but a metrics table with uncertainty and a
calibration check.

## Tasks

1. Train (or reuse) a classifier on your dataset. On the validation set compute overall accuracy, per-class
   precision and recall, and a confusion matrix.
2. **Error bars.** Attach a 95% confidence interval to the overall accuracy (Wilson or Clopper–Pearson). State
   how many test images you would need to distinguish two models that differ by 1 point.
3. Compute macro- and micro-averaged F1 and explain when they diverge (hint: class imbalance). If your dataset
   has many classes, add top-3 accuracy.
4. **Calibration.** Plot a reliability diagram (accuracy vs. confidence, binned) and compute the Expected
   Calibration Error. Then fit a temperature `T` on the validation set, apply `z/T`, and report ECE before and
   after — confirming accuracy is unchanged.
5. From the confusion matrix, name the two most-confused class pairs and hypothesize *why* (visual similarity,
   few examples, label noise).

## Deliverable

A notebook producing a metrics table (with the accuracy CI), a confusion matrix, a reliability diagram with ECE
before/after temperature scaling, and a short written error analysis naming the confused pairs and a hypothesis
for each. The analysis and the calibration matter more than the accuracy number.
