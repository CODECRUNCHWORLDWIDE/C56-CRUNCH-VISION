# Exercise 2 — Train and honestly evaluate a CNN

**Goal:** a working, correctly-regularized, honestly-evaluated CNN.

## Tasks

1. Load MNIST or CIFAR-10 with torchvision, **normalized with the training-set channel statistics**, in
   train/test DataLoaders (use `num_workers` for speed).
2. Build a small CNN: three conv->BatchNorm->ReLU->pool blocks (channels 3->64->128->256), then global
   average pool and a linear head. Print the shape after every block once to confirm your arithmetic.
3. Train with `CrossEntropyLoss` and SGD-with-momentum (lr 0.1, momentum 0.9, weight decay 5e-4) plus a
   cosine LR schedule, for enough epochs to converge. Log training and validation accuracy each epoch and
   plot both curves.
4. **Sanity check first:** before the full run, overfit a *single batch* to ~100% accuracy to prove the
   loss, gradients, and shapes are wired correctly. Note it in the notebook.
5. Report final test accuracy and draw a confusion matrix. Identify the two most-confused classes and show
   a few misclassified images.

## Deliverable

A notebook with the single-batch overfit check, train/val curves, final test accuracy, and a confusion
matrix. If train accuracy climbs but validation stalls, you are overfitting — note it; Exercise 3 quantifies
the fix.
