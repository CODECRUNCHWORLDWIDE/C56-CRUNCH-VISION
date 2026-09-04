# Exercise 1 — Shapes, receptive fields, and FLOPs by the numbers

**Goal:** never guess a CNN's tensor shapes, receptive field, or cost again.

## Part A — output shapes (paper, then verify)

1. Using the formula `W_out = floor((W + 2p - d*(k-1) - 1)/s) + 1`, compute the output shape of
   `Conv2d(in=3, out=16, kernel=3, padding=1, stride=1)` on `(N, 3, 32, 32)`. Verify in PyTorch.
2. Repeat for `padding=0`; then `stride=2`; then `dilation=2, padding=2`. Predict each shape *before*
   running it.
3. Add `MaxPool2d(2, 2)` after a conv and compute the shape through both layers.
4. Build a 3-block stack (conv->relu->pool x3) on 32x32 and print the shape after every layer; confirm
   spatial size shrinks and channels grow as designed.

## Part B — receptive field

5. Using the recurrence `r += (k-1)*j; j *= s` (init `r=1, j=1`), compute the theoretical receptive field
   of a stack of three `3x3` stride-1 convs. Then insert a `2x2` stride-2 pool after each conv and recompute
   — show how pooling accelerates RF growth.

## Part C — FLOPs

6. Using `C_in * C_out * k * k * H' * W'`, compute the multiply-add count of the first conv of your Part A
   stack. Compare it to the cost of a `1x1` conv with the same channel counts, and state the ratio.

## Deliverable

A notebook where every predicted shape and RF value (written in a comment) matches PyTorch's actual output,
plus the FLOP comparison. Shape and cost fluency prevents most CNN bugs.
