# Lecture 4 — Backpropagation through convolution: the math and the im2col trick

Every introductory course tells you "backprop handles the gradients." A graduate course derives
them. This lecture works out the backward pass of a convolution from scratch — you will see that the
gradient of a convolution is *itself* a convolution, understand why frameworks compute it via a single
matrix multiply (`im2col`), and be ready to implement and gradient-check it in Exercise 4.

## Convolution as a linear operator

Fix the notation. A single-channel 2-D convolution (frameworks compute cross-correlation) of input `X` with
kernel `K` of size `k x k` produces output `Y`:

    Y[i, j] = sum_{a=0}^{k-1} sum_{b=0}^{k-1} K[a, b] * X[i + a, j + b].

Because this is linear in both `X` and `K`, it can be written `y = A x` for a fixed sparse matrix `A` (a
doubly-block-Toeplitz matrix built from `K`). Holding that view in mind makes the gradients fall out of
ordinary matrix calculus.

## The three gradients

Let `L` be the scalar loss and `dY = dL/dY` the upstream gradient flowing into the layer's output (same
shape as `Y`). We need three things.

**1. Gradient w.r.t. the kernel `K`.** By the chain rule, `dL/dK[a,b] = sum_{i,j} dY[i,j] * dX_term`, and
since `dY[i,j]/dK[a,b] = X[i+a, j+b]`,

    dL/dK[a, b] = sum_{i, j} dY[i, j] * X[i + a, j + b].

Read the right-hand side: it is the **cross-correlation of the input `X` with the upstream gradient `dY`**.
The kernel gradient is a convolution.

**2. Gradient w.r.t. the input `X`.** Each input pixel `X[p,q]` influences every output `Y[i,j]` for which
`(a,b) = (p-i, q-j)` is a valid kernel index. Summing contributions:

    dL/dX[p, q] = sum_{i, j} dY[i, j] * K[p - i, q - j].

This is a **full convolution of the upstream gradient `dY` with the kernel flipped 180 degrees** (`K[p-i,
q-j]` is the flipped kernel). So: forward pass is correlation with `K`; input-gradient is convolution with
`flip(K)`. This is the concrete payoff of the "cross-correlation vs. convolution" caveat from Lecture 1 —
the flip that was cosmetic in the forward pass is *load-bearing* in the backward pass.

**3. Gradient w.r.t. the bias.** With one bias per output channel, `dL/db = sum_{i,j} dY[i,j]` — just sum
the upstream gradient over spatial positions (and over the batch).

Multi-channel case: sum the kernel-gradient over the batch, and the input-gradient over output channels;
the structure is identical, just with channel index bookkeeping.

## im2col: turning convolution into one matmul

Naive nested loops are correct but slow. The standard implementation — the one behind cuDNN's baseline and
Caffe (Jia et al., 2014) — is **im2col**: unfold every `k x k x C_in` receptive-field patch of the input
into a column, stacking them into a matrix `X_col` of shape `(k*k*C_in, H'*W')`. Reshape the kernels into a
matrix `W` of shape `(C_out, k*k*C_in)`. Then

    Y_flat = W @ X_col,   shape (C_out, H'*W'),

and a reshape gives `Y`. The convolution becomes a single dense matrix multiply — exactly what GPUs execute
near their FLOP ceiling. The backward pass reuses the same machinery: `dW = dY_flat @ X_col^T`,
`dX_col = W^T @ dY_flat`, and a **col2im** operation (the transpose of im2col, which *scatter-adds*
overlapping patches back) turns `dX_col` into `dX`. That scatter-add is precisely the "full convolution
with the flipped kernel" from derivation (2), viewed through the matrix lens.

The trade-off: im2col materializes overlapping patches, using `~k*k` times the input memory. Alternatives —
FFT-based convolution (fast for large kernels; convolution is pointwise multiply in the Fourier domain) and
Winograd minimal-filtering algorithms (Lavin & Gray, 2016), which cut the multiply count for small kernels
like `3x3` by ~2.25x — power production libraries. But im2col is the one to understand first because the
gradients are transparent.

## Computational complexity

A conv layer's forward FLOPs are approximately

    C_in * C_out * k * k * H' * W'    (multiply-adds).

Note what dominates: for a fixed feature-map size, cost is linear in both channel counts and quadratic in
kernel size. This is why `1x1` convolutions (Lin et al., 2014) — pure channel mixing, `k=1` — are so cheap
and ubiquitous (bottlenecks in ResNet, Inception), and why depthwise-separable convolutions (Lecture 5)
factor the `C_in * C_out` term to make mobile nets affordable.

## Gradient checking: your safety net

Never trust a hand-derived backward pass without a **numerical gradient check**. For each parameter `w`,
compare the analytic gradient to the central finite difference

    dL/dw ~= ( L(w + h) - L(w - h) ) / (2h),   h ~ 1e-5,

and require the relative error `|analytic - numeric| / (|analytic| + |numeric| + 1e-12)` to be below ~1e-5
(central differences are `O(h^2)` accurate). Exercise 4 has you implement conv forward and backward from
scratch and pass exactly this check against `torch.autograd`. If it passes, you *own* the operation.

## Common pitfalls

- **Forgetting the flip.** Input-gradient is a *full convolution with the flipped kernel*; skip the flip and
  your gradients are subtly wrong and the check fails.
- **Padding bookkeeping in col2im.** Overlapping patches must be *accumulated* (scatter-add), not
  overwritten; overwriting silently drops gradient mass.
- **One-sided differences for the check.** Use *central* differences; one-sided is `O(h)` and gives false
  failures near the tolerance.

**Takeaway:** the backward pass of a convolution is another convolution — kernel-gradient is the correlation
of input with upstream gradient, input-gradient is a full convolution with the 180-degree-flipped kernel,
bias-gradient is a spatial sum. Frameworks realize all of this as `im2col` matmuls plus a col2im
scatter-add, and you validate any implementation with a central-difference gradient check.
