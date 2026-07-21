# Week 3 — Convolutional neural networks for vision

> **Goal:** by Sunday you can explain why convolution beats fully-connected layers for images, implement a conv layer and pooling, read a feature map, and train a small CNN on MNIST or CIFAR-10 that clearly beats a linear baseline.

Week 1 you convolved images with kernels you *chose*. Week 2 you engineered features by hand. This week the network **learns the kernels itself**. A convolutional neural network (CNN) is the natural architecture for images: it shares weights across the image, respects spatial locality, and builds a hierarchy from edges to textures to objects. You'll implement the pieces — convolution, pooling, channels — then train a real CNN, leaning on your [C53 Crunch Nets](../C53-CRUNCH-NETS/) backprop knowledge to know exactly what's being optimized.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** why a fully-connected network is wrong for images and how convolution fixes it (weight sharing, locality).
- **Implement** a convolution layer and pooling, and compute output shapes for any config.
- **Reason** about channels, receptive fields, and how depth builds a feature hierarchy.
- **Build** a small CNN in PyTorch and train it on MNIST or CIFAR-10.
- **Visualize** learned filters and feature maps to see what the network responds to.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Why convolution, not fully-connected](lecture-notes/01-why-convolution.md)
2. [Lecture 2 — Pooling, channels & receptive fields](lecture-notes/02-pooling-channels-receptive-fields.md)
3. [Lecture 3 — Training a CNN that works](lecture-notes/03-training-a-cnn.md)

**Exercises**

1. [Exercise 1 — Convolution and pooling by the numbers](exercises/exercise-01-conv-output-shapes.md)
2. [Exercise 2 — Train a CNN on MNIST or CIFAR-10](exercises/exercise-02-train-a-cnn.md)
3. [Exercise 3 — Augmentation and the dense baseline](exercises/exercise-03-augmentation-and-baseline.md)

**Challenges**

1. [Challenge 1 — See what your network learned](challenges/challenge-01-visualize-filters.md)
2. [Challenge 2 — Find what actually matters](challenges/challenge-02-architecture-hunt.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 4.

## Deliverable

A small CNN trained in PyTorch on MNIST or CIFAR-10 that beats a fully-connected baseline, with a training/validation curve, a confusion matrix, and a visualization of the first-layer learned filters.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
