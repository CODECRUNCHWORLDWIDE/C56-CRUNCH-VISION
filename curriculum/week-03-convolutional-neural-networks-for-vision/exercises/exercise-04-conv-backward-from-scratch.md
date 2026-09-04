# Exercise 4 — Implement convolution's forward and backward pass, and gradient-check it

**Goal:** *own* the convolution operator by implementing its forward and backward passes from
scratch and passing a numerical gradient check against autograd. This is the deliverable that turns Lecture
4 from reading into knowledge.

## Part A — forward (NumPy or torch tensors, no autograd)

1. Implement `conv2d_forward(X, K, b, stride=1, pad=0)` for a single image with `C_in` input channels,
   `C_out` kernels of size `k x k`, using the cross-correlation definition
   `Y[o,i,j] = b[o] + sum_c sum_{a,b'} K[o,c,a,b'] * X[c, i*s+a, j*s+b']`.
2. Verify it matches `torch.nn.functional.conv2d` on random inputs with `torch.allclose` (atol 1e-5) for
   several strides and paddings.

## Part B — backward (derive, then implement)

3. Implement `conv2d_backward(dY, X, K)` returning `dK`, `dX`, and `db`, using the three results from
   Lecture 4: `dK` = correlation of `X` with `dY`; `dX` = full convolution of `dY` with the **180-degree
   flipped** kernel; `db` = spatial sum of `dY`. Handle the multi-channel bookkeeping.

## Part C — gradient check

4. Build a scalar loss `L = sum(Y * G)` for a fixed random `G` (so `dY = G`). Compute analytic gradients with
   your backward pass, and numeric gradients with *central* differences `(L(w+h) - L(w-h)) / (2h)`,
   `h = 1e-5`, for a handful of entries of `K`, `X`, and `b`. Require relative error < 1e-5.
5. Cross-check against `torch.autograd`: wrap the same inputs with `requires_grad=True`, call `.backward()`,
   and confirm your `dK`, `dX`, `db` match to 1e-5.

## Deliverable

A notebook where (i) your forward matches `F.conv2d`, (ii) your central-difference check passes at < 1e-5
relative error, and (iii) your analytic gradients match autograd. Include one sentence explaining why the
input-gradient needs the kernel *flip*.
