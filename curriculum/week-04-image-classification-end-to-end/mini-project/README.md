# Mini-Project — An End-to-End Image Classification Pipeline

## Brief

Take a real image dataset from raw folders to a trained, honestly-evaluated classifier — the complete pipeline you'll reuse for the capstone. The model can be modest; the *pipeline and evaluation* are what's graded.

## Requirements

1. **Data:** a real multi-class dataset (Oxford Pets subset, a folder dataset, or your own). Custom `Dataset`/`DataLoader`, train/val/test splits with no leakage (mind duplicates and groups), train-only normalization stats, and augmented training transforms.
2. **Model:** a CNN (yours from Week 3, or a small standard architecture) with regularization — augmentation, weight decay, and/or dropout.
3. **Training:** an LR schedule, early stopping on validation, and logged train/validation curves.
4. **Imbalance:** if the data is imbalanced, handle it explicitly and report per-class metrics.
5. **Evaluation:** on the **test set touched once**, report accuracy, per-class precision/recall, a confusion matrix, and (for many classes) top-k. Write an error analysis naming the failure modes with example misclassified images.
6. **README:** how to reproduce, and an honest limitations section.

## Stretch

- Add test-time augmentation (average predictions over a few augmented views) and measure the gain.
- Compare SGD+momentum+cosine vs. AdamW for final accuracy.

## What you're proving

You can run a real classification project end to end — data, training, and honest evaluation — and diagnose it when it misbehaves. This is the exact shape of the capstone. Next week you stop training from scratch and stand on pretrained backbones, which is how nearly all real vision is done.
