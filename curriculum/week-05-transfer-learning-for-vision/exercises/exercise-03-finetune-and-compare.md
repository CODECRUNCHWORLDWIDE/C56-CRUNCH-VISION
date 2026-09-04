# Exercise 3 — Fine-tune with discipline and compare strategies

**Goal:** measure the feature-extraction vs. fine-tuning trade-off yourself, with the responsible
recipe from Lecture 3.

## Tasks

1. Starting from your trained head (Exercise 2 — i.e. LP-FT), unfreeze the last backbone block and fine-tune
   with a **small, warmed-up, cosine-decayed** learning rate (10-100x smaller than from-scratch). Add
   **layer-wise LR decay** (a per-layer multiplier `xi < 1` from the head down) and report its effect.
2. Use augmentation, weight decay, and early stopping. Track the train/validation curves.
3. Compare against Exercise 2's feature-extraction result on **accuracy, training time, and overfitting gap**.
   Assemble a clean comparison table.
4. **Catastrophic-forgetting demo.** Fine-tune once with a deliberately *too-large* LR and no warmup, unfreezing
   everything on step one. Show accuracy drops below the frozen extractor, and explain — in Lecture-3 terms —
   what happened to the pretrained features.
5. **LP-FT vs. FT-from-random.** Fine-tune once starting from a *random* head (no linear-probe stage) and once
   with LP-FT. Compare in-distribution accuracy and, if you have any shifted/held-out-hard set,
   out-of-distribution accuracy; relate to Kumar et al. (2022).

## Deliverable

A notebook with the strategy comparison table (feature extraction vs. fine-tune: accuracy, time, gap), the
catastrophic-forgetting demonstration, the LP-FT vs. FT-from-random comparison, and a written recommendation of
which strategy fits your dataset and why.
