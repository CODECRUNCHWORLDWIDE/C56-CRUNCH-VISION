# Exercise 3 — Augmentation and the dense baseline

**Goal:** prove convolution's advantage and augmentation's value.

## Tasks

1. **Baseline:** train a fully-connected network (flatten → dense → dense → softmax) on the same dataset with comparable parameter budget. Record its test accuracy.
2. Compare it to your CNN from Exercise 2. Quantify how much convolution buys you.
3. **Augmentation:** add random crop + horizontal flip to the CNN's *training* transforms only, retrain, and compare the train/validation gap and final accuracy to the un-augmented CNN.
4. Explain why augmentation shrinks the overfitting gap.

## Deliverable

A notebook with a table: dense baseline vs. CNN vs. CNN+augmentation (accuracy and train/val gap). Two sentences on what convolution and augmentation each contributed. The numbers should make the case for both.
