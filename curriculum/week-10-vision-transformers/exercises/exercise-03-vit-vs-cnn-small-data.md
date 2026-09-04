# Exercise 3 — ViT vs. CNN under a controlled data budget

**Goal:** experience the inductive-bias / data-hunger trade firsthand, under a fair protocol.

## Tasks

1. On a small dataset (a few hundred images, a handful of classes), fine-tune a pretrained ViT and a
   pretrained CNN of comparable parameter count **identically**: same augmentation, LR schedule, epochs,
   optimizer. Report held-out accuracy, wall-clock training time, and the train-minus-val overfitting gap.
2. **The recipe confound (ConvNeXt lesson).** Now vary the augmentation strength (light vs. heavy
   RandAugment/Mixup) for *both* models and show how much of any gap is the recipe, not the architecture.
   Hold everything else fixed.
3. **From scratch.** Train both *without* pretraining on the same small data. Which degrades more? Connect
   the ViT's larger collapse directly to its weaker inductive biases (Lecture 3).
4. **Data-scale sweep.** Subsample the training set to 25%, 50%, 100% and plot accuracy vs. data fraction
   for both models. The curves should cross or converge as data grows — the data-hunger story as a picture.
5. Summarize which model you would ship for this dataset and defend it with the numbers.

## Deliverable

A notebook with a comparison table (pretrained-finetune and from-scratch, ViT vs. CNN: accuracy, time, gap),
an accuracy-vs-data-fraction plot, and a justified recommendation. The from-scratch collapse of the ViT on
small data — and its recovery with more data — is the lesson to see with your own eyes.
