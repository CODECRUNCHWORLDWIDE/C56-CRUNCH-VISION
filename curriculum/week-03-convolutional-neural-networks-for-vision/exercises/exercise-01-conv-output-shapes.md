# Exercise 1 — Convolution and pooling by the numbers

**Goal:** never guess a CNN's tensor shapes again.

## Tasks

1. For a `Conv2d(in=3, out=16, kernel=3, padding=1, stride=1)` on a `(N, 3, 32, 32)` input, compute the output shape by hand using the size formula, then verify in PyTorch.
2. Repeat for `padding=0`, then for `stride=2`. Predict each output shape *before* running it.
3. Add a `MaxPool2d(2, 2)` after a conv and compute the shape through both layers.
4. Build a 3-block conv stack (conv→relu→pool ×3) on a 32×32 input and print the shape after every layer. Confirm spatial size shrinks and channels grow as designed.

## Deliverable

A notebook where every predicted shape (written in a comment) matches PyTorch's actual output. Shape fluency prevents most CNN bugs.
