# Mini-Project — An End-to-End, Honestly-Evaluated Image Classifier

## Brief

Take a real image dataset from raw folders to a trained, honestly-evaluated classifier — the complete pipeline
you will reuse for the capstone. The model can be modest; the **pipeline, the evaluation, and the honesty** are
what is graded. Every design decision should be defensible from the theory of Lectures 1–5.

## Requirements

1. **Data (Lecture 1).** A real multi-class dataset (Oxford-IIIT Pets subset, a folder dataset, or your own).
   A custom `Dataset`/`DataLoader`, **group-aware** train/val/test splits with a programmatic leakage audit
   (no image *and* no group spans two splits), **train-only** normalization statistics, and augmented training
   transforms. State what the group unit is.
2. **Model & regularization (Lectures 2–4).** A CNN (yours from Week 3, or a small standard architecture) with
   deliberate regularization — augmentation, weight decay, dropout, and/or label smoothing — each justified as a
   bias/variance trade.
3. **Training (Lectures 3–4).** An LR schedule (cosine or step, with warmup if you increase batch size), a
   sensible optimizer (SGD+momentum or AdamW — use AdamW, not Adam+weight_decay), early stopping on validation,
   and logged train/validation curves.
4. **Imbalance (Lecture 5).** If the data is imbalanced, handle it explicitly (class weighting, resampling, or
   focal loss), report per-class recall before/after, and report **balanced accuracy**.
5. **Evaluation (Lectures 2, 5), test set touched once.** Report overall accuracy **with a confidence
   interval**, per-class precision/recall, a confusion matrix, and (for many classes) top-k. Produce a
   **reliability diagram with ECE before and after temperature scaling**. Write an error analysis naming the
   failure modes with example misclassified images, and — if the data touches people or has an obvious subgroup
   attribute — disaggregate metrics by subgroup.
6. **README.** How to reproduce, and an honest limitations section stating what the reported number does and
   does not establish (test-set size, distribution assumptions, known failure modes).

## Stretch

- Add test-time augmentation (average predictions over a few augmented views) and measure the gain against the
  accuracy confidence interval — is it real or noise?
- Add a reject option using calibrated confidence and plot the risk–coverage curve.
- Compare SGD+momentum+cosine vs. AdamW for final accuracy and calibration.

## What you are proving

You can run a real classification project end to end — data, training, and *honest* evaluation with uncertainty
and calibration — and diagnose it when it misbehaves. This is the exact shape of the capstone. Next week you
stop training from scratch and stand on pretrained backbones, which is how nearly all real vision is done.
