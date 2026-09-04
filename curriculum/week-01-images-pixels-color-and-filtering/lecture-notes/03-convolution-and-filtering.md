# Lecture 3 — Convolution: the impulse response of a linear shift-invariant system

Convolution is the operation that runs from the oldest image filters to the deepest CNN. The
introductory framing — "slide a kernel, take a weighted sum" — is operationally correct but conceptually
thin. The graduate framing is: **convolution is the unique operation performed by a linear shift-invariant
(LSI) system, and the kernel is that system's impulse response.** Once you see filtering this way, the
convolution theorem, separability, and the design of every filter become consequences rather than tricks
(Gonzalez & Woods, *Digital Image Processing*, 4th ed., 2018, Ch. 3).

## Linearity + shift-invariance forces convolution

A filter is an operator `T` mapping an input image to an output. Two properties define the class we care
about:

- **Linearity:** `T(a f + b g) = a T(f) + b T(g)`.
- **Shift-invariance:** shifting the input shifts the output identically — `T` has no preferred location.

A foundational result of linear systems theory: **any** operator that is both linear and shift-invariant can
be written as a convolution with a fixed kernel `h`, the operator's response to an impulse (a single bright
pixel). In 2-D, discrete:

    (f * h)[m, n] = Σ_i Σ_j f[m − i, n − j] h[i, j].

So designing a filter *is* choosing an impulse response. The kernel's values are the whole filter.

## Sliding-window implementation

Operationally, convolution slides the kernel over every position, multiplies overlapping values, and sums:

```python
import numpy as np
def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = (region * kernel).sum()
    return out
```

That triple loop is slow but *exactly* what a convolution layer computes, minus the optimized C/GPU
underneath.

## Convolution vs. correlation — the flip

True mathematical convolution **flips** the kernel (note the `f[m−i, n−j]` indices) before sliding; what most
libraries and every CNN compute is **cross-correlation** (no flip, `f[m+i, n+j]`). For symmetric kernels
(box, Gaussian) the two agree; for learned kernels the network simply learns the flipped weights, so everyone
says "convolution" and means correlation. Know the distinction so a textbook's flip never confuses you — and
so you know convolution (unlike correlation) is commutative and associative, which is what lets you compose
filters.

## What different kernels do

- **Box blur** — all `1/9` (3×3): averages each neighborhood, smoothing noise and detail alike; its frequency
  response is a sinc, which ripples (ringing).
- **Gaussian blur** — a bell-shaped kernel `h(i,j) ∝ exp(−(i²+j²)/2σ²)`; the *optimal* smoother in a
  joint space/frequency sense (the Gaussian uniquely minimizes the uncertainty product), no ringing. The
  standard denoiser; `σ` sets the scale.
- **Sharpen** — `[[0,−1,0],[−1,5,−1],[0,−1,0]]`: identity plus a Laplacian, boosting the center against its
  neighbors (unsharp masking).
- **Sobel (gradient)** — `[[−1,0,1],[−2,0,2],[−1,0,1]]`: a smoothed horizontal derivative — an edge detector,
  the seed of Week 2.

One operation, different weights, gives blurring, sharpening, and edge finding. A CNN merely *learns* the
weights instead of you choosing them.

## Separability: a k×k Gaussian is two 1-D passes

A kernel is **separable** if it factors as an outer product `h[i,j] = a[i] b[j]`. The Gaussian is separable
because `exp(−(i²+j²)/2σ²) = exp(−i²/2σ²)·exp(−j²/2σ²)`. Then `f * h = (f * a) * b` — convolve rows with a
1-D `a`, then columns with `b`. Cost drops from `k²` multiplies per pixel to `2k`; for a 15×15 Gaussian that
is 225 vs 30, a 7.5× win. Separability is a low-rank (rank-1) property of the kernel matrix; the box filter is
separable too. This is the first of many places where structure buys speed.

## Padding, stride, and output size

- **Padding** adds a border (zeros or edge-replicated) so the output keeps size and border pixels are not
  lost; zero-padding darkens edges, edge-replication ("clamp") avoids it.
- **Stride** is how far the kernel jumps; stride 2 halves resolution — a cheap downsample (but see Lecture 4:
  stride without a pre-blur aliases).
- Output size for a `k×k` kernel, padding `p`, stride `s` on input `n`: `floor((n + 2p − k)/s) + 1`. You will
  use this formula constantly designing networks.

**Takeaway:** convolution is not a trick — it is the *only* thing a linear shift-invariant system can do, and
the kernel is its impulse response. That framing gives you the flip (convolution vs. correlation),
composition (associativity), separability (a Gaussian = two 1-D passes, `2k` vs `k²`), and the output-size
formula `floor((n+2p−k)/s)+1`. CNNs are this same operation with *learned* impulse responses.
