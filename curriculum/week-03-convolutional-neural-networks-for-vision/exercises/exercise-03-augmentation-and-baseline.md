# Exercise 3 — The dense baseline, augmentation, and an ablation

**Goal:** prove convolution's advantage and measure each regularizer's contribution — with
controlled experiments, not folklore.

## Tasks

1. **Baseline.** Train a fully-connected network (flatten -> dense -> dense -> softmax) on the same dataset
   with a *comparable parameter budget* to your CNN. Record test accuracy. State the parameter counts of
   both so the comparison is fair.
2. **The convolution gap.** Compare the dense baseline to your Exercise 2 CNN. Quantify how much convolution
   buys you, and argue why (tie to the equivariance/weight-sharing prior from Lecture 1).
3. **Augmentation ablation.** Add `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` to the CNN's
   *training* transforms only, retrain, and compare the train/validation gap and final accuracy to the
   un-augmented CNN.
4. **BN ablation.** Remove batch norm from the CNN, retrain, and report the effect on training stability and
   final accuracy. Keep every other knob fixed — change one thing at a time.
5. Explain, in terms of Lecture 3, why augmentation shrinks the overfitting gap (injected label invariances)
   and why BN helps (smoother loss landscape).

## Deliverable

A notebook with a table — dense baseline vs. CNN vs. CNN+aug vs. CNN-without-BN (accuracy, parameter count,
train/val gap) — and two sentences on what convolution, augmentation, and BN each contributed. The numbers
should make the case; note where any difference is within run-to-run noise.
