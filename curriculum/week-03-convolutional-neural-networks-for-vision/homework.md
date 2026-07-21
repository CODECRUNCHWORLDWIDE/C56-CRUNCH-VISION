# Week 3 — Homework

Cement the CNN before scaling it into a full pipeline next week.

## Tasks

- Derive, in writing, why weight sharing makes a conv layer's parameter count independent of image size.
- Compute by hand the output shape of a 4-layer conv stack you design, then verify in PyTorch.
- Add batch normalization to your mini-project CNN and report the change in training speed and accuracy.
- Read the CS231n Convolutional Networks notes (in resources) and list two things about receptive fields you didn't know.

## Definition of done

A committed CNN classifier on MNIST or CIFAR-10 that beats a fully-connected baseline, trained with augmentation, reporting train/validation curves, a confusion matrix, final held-out accuracy, and a visualization of first-layer learned filters — with a short note interpreting the errors.

Submit by committing your work to your course repo under `week-03/`.
