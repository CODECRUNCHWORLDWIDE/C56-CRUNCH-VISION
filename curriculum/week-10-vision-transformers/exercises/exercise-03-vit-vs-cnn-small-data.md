# Exercise 3 — ViT vs. CNN on small data

**Goal:** experience the data-hunger trade-off firsthand.

## Tasks

1. On a small dataset (a few hundred images, few classes), fine-tune a pretrained ViT and a pretrained CNN of comparable size, identically (same augmentation, LR schedule, epochs).
2. Compare held-out accuracy, training time, and the overfitting gap.
3. Now try training *both from scratch* (no pretraining) on the same small data. Which degrades more? Connect to the ViT's weaker inductive biases.
4. Summarize which you'd ship for this dataset and why.

## Deliverable

A notebook with a comparison table (pretrained fine-tune and from-scratch, ViT vs. CNN: accuracy, time, gap) and a justified recommendation. The from-scratch collapse of the ViT on small data is the lesson to see with your own eyes.
