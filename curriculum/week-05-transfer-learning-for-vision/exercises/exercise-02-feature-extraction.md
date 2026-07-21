# Exercise 2 — Train a frozen feature extractor

**Goal:** transfer learning the fast, small-data-safe way.

## Tasks

1. Freeze all backbone parameters (`requires_grad = False`); leave only the new head trainable.
2. Train just the head on your small dataset with augmentation. This should be *fast* — only a small classifier is learning.
3. Report validation accuracy and the train/validation gap. Note how little it overfits.
4. Time the training and compare to what training a CNN from scratch (Week 3) took for similar accuracy.

## Deliverable

A notebook training the frozen-backbone model, with accuracy, the train/val gap, and the training time. Two sentences on why this barely overfits even on small data (the backbone can't memorize your images).
