# Week 3 — Convolutional neural networks for vision

> **Goal:** by Sunday you can (1) state precisely why a convolution is the right linear map for images — translation equivariance, weight sharing, and locality — and quantify the parameter and FLOP savings over a dense layer; (2) do receptive-field arithmetic and distinguish the theoretical from the effective receptive field; (3) derive and implement the backward pass of a convolution from scratch and pass a numerical gradient check; and (4) build, train, and honestly evaluate a CNN that beats a dense baseline, then explain the design lineage from LeNet to ResNet in terms of the problems each architecture solved.

Week 1 you convolved images with kernels you *chose*; Week 2 you engineered features by hand. This week the network **learns the kernels itself**, and we treat the convolutional neural network (CNN) not as a bag of tricks but as a principled answer to a precise question: what is the right *linear operator* for data that lives on a grid and whose statistics are translation-invariant?

The answer — convolution — is forced, not chosen. If you demand a linear layer that is **equivariant to translation** (shift the input, the output shifts identically) and **local** (each output depends on a bounded neighborhood), the only maps that qualify are convolutions with compact support. Weight sharing is then a *consequence*, not a separate assumption, and it collapses a 150-million-parameter dense layer into a 27-weight kernel. We make this exact, count the parameters and FLOPs, and then go where introductory courses stop: we **derive the backward pass** of convolution (it is itself a convolution — a correlation with the flipped kernel), implement it from scratch via `im2col`, and gradient-check it. We treat normalization and augmentation as *regularization theory*, not folklore, and we close with the architectural lineage — LeNet (LeCun et al., 1998), AlexNet (Krizhevsky et al., 2012), VGG (Simonyan & Zisserman, 2015), GoogLeNet (Szegedy et al., 2015), and ResNet (He et al., 2016) — reading each as the solution to a specific optimization or generalization failure of its predecessor.

By Sunday you will have trained a real CNN that beats a dense baseline, but more importantly you will understand *why* it must, at a level of rigor the next nine weeks stand on.

## Learning objectives

By the end of this week, you will be able to:

- **Prove** that translation equivariance plus locality forces a linear image layer to be a convolution, and derive weight sharing as a consequence rather than a separate assumption.
- **Quantify** the parameter and FLOP cost of conv vs. dense layers, and compute output shapes for any (kernel, stride, padding, dilation) configuration without guessing.
- **Compute** the theoretical receptive field of a deep stack via the recurrence, and explain why the *effective* receptive field is Gaussian and far smaller (Luo et al., 2016).
- **Derive** the backward pass of a convolution — showing the gradient w.r.t. the input is a full convolution with the flipped kernel — and implement it from scratch with a passing numerical gradient check.
- **Explain** batch normalization, dropout, and label-preserving augmentation as distinct regularizers, and state what each assumes and what failure mode each targets.
- **Build** a CNN in PyTorch, train it with the correct data pipeline, and evaluate it honestly with train/val curves, a confusion matrix, and error analysis.
- **Trace** the architectural lineage LeNet→AlexNet→VGG→GoogLeNet→ResNet, naming the specific failure each design fixed, and articulate why residual connections make very deep networks trainable.
- **Diagnose** a CNN that fails to beat a dense baseline by isolating the cause — normalization, augmentation, a shape bug, or an optimization failure — from evidence, not folklore.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 231N` — explain the architecture, forward pass and backward pass of a convolutional neural network, and train one on an image dataset. |
| Industry | Establish that a convolutional model beats the simpler baseline it is meant to replace, and show the training and validation curves that prove it rather than quoting a single number. |
| Beyond the bar | implements the convolution backward pass from scratch and holds it to a numerical gradient check against autograd — `exercises/exercise-04-conv-backward-from-scratch.md` |

## Prerequisites

- Week 1 (image formation, convolution with hand-chosen kernels) and Week 2 (classical features).
- The C53 Crunch Nets training loop: forward, loss, backward, optimizer step; cross-entropy; SGD/Adam.
- Linear algebra: matrix multiplication, Toeplitz/banded structure, and the idea of a linear operator.
- Multivariable calculus and the chain rule (for the convolution backward-pass derivation).

## This week

**Lectures**

1. [Lecture 1 — Why convolution, not fully-connected: equivariance forces the operator](lecture-notes/01-why-convolution.md)
2. [Lecture 2 — Pooling, channels, and the receptive field (theoretical vs. effective)](lecture-notes/02-pooling-channels-receptive-fields.md)
3. [Lecture 3 — Training a CNN that works: normalization and augmentation as regularization](lecture-notes/03-training-a-cnn.md)
4. [Lecture 4 — Backpropagation through convolution: the math and the im2col trick](lecture-notes/04-backprop-through-convolution.md)
5. [Lecture 5 — The architectural lineage: from LeNet to ResNet, and why depth needed residuals](lecture-notes/05-modern-cnn-architectures.md)

**Exercises**

1. [Exercise 1 — Shapes, receptive fields, and FLOPs by the numbers](exercises/exercise-01-conv-output-shapes.md)
2. [Exercise 2 — Train and honestly evaluate a CNN](exercises/exercise-02-train-a-cnn.md)
3. [Exercise 3 — The dense baseline, augmentation, and an ablation](exercises/exercise-03-augmentation-and-baseline.md)
4. [Exercise 4 — Implement convolution's forward and backward pass, and gradient-check it](exercises/exercise-04-conv-backward-from-scratch.md)

**Challenges**

1. [Challenge 1 — See what your network learned](challenges/challenge-01-visualize-filters.md)
2. [Challenge 2 — Find what actually matters (controlled ablation)](challenges/challenge-02-architecture-hunt.md)
3. [Challenge 3 — Reproduce the degradation problem and fix it with residuals](challenges/challenge-03-degradation-and-residuals.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 4.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A CNN trained in PyTorch on MNIST or CIFAR-10 that beats a fully-connected baseline, shipped with training/validation curves, a confusion matrix with error analysis, and a visualization of the first-layer learned filters — plus a from-scratch convolution forward/backward pass that passes a numerical gradient check against autograd.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
