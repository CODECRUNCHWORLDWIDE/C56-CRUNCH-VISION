# Exercise 3 — Fine-tune and compare strategies

**Goal:** measure the feature-extraction vs. fine-tuning trade-off yourself.

## Tasks

1. Starting from your trained head (Exercise 2), unfreeze the last backbone block and fine-tune with a **small** learning rate (e.g. 10–100× smaller than from-scratch).
2. Use augmentation and early stopping. Track the train/validation curves.
3. Compare against Exercise 2's feature-extraction result: accuracy, training time, and overfitting gap.
4. Deliberately fine-tune once with a *too-large* LR and observe catastrophic forgetting (accuracy drops below the frozen extractor). Explain what happened.

## Deliverable

A notebook with a comparison table (feature extraction vs. fine-tune: accuracy, time, gap) and the catastrophic-forgetting demonstration. A written recommendation of which strategy fits your dataset and why.
