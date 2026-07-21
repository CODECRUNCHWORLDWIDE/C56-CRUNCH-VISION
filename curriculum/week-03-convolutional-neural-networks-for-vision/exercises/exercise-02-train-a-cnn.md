# Exercise 2 — Train a CNN on MNIST or CIFAR-10

**Goal:** a working, honestly-evaluated CNN.

## Tasks

1. Load MNIST or CIFAR-10 with torchvision, normalized, in train/test DataLoaders.
2. Build a small CNN: two or three conv+ReLU+pool blocks, then a dense classifier head with softmax over the classes.
3. Train with `CrossEntropyLoss` and Adam for several epochs. Log training and validation accuracy each epoch and plot both curves.
4. Report final test accuracy and draw a confusion matrix. Identify the two most-confused classes.

## Deliverable

A notebook with the training/validation curves, final test accuracy, and a confusion matrix. If your training accuracy climbs but validation stalls, you're overfitting — note it; Exercise 3 fixes it.
