# Lecture 1 — Optical flow: the motion field

**Optical flow** is the dense field of apparent per-pixel motion between two consecutive
frames: a vector `(u, v)` at every pixel saying "this point moved *here*." It is the lowest-level
description of motion and it underpins video compression, stabilization, slow-motion interpolation,
and motion-based recognition. This lecture derives it, exposes why it is fundamentally ambiguous, and
closes that ambiguity three ways.

## Brightness constancy and the flow constraint

Flow rests on one assumption: a physical point keeps roughly the **same brightness** as it moves.
If the pixel at `(x, y)` with intensity `I` moves by `(u, v)` in time `dt`:

    I(x, y, t) = I(x + u·dt, y + v·dt, t + dt)

A first-order Taylor expansion of the right-hand side about `(x, y, t)` gives, after cancelling `I`
and dividing by `dt`, the **optical flow constraint equation** (Horn & Schunck 1981):

    I_x·u + I_y·v + I_t = 0        (equivalently  ∇I · (u,v) + I_t = 0)

where `I_x, I_y` are spatial gradients (Week 1's Sobel) and `I_t` is the temporal gradient (a frame
difference). This is **one** scalar equation with **two** unknowns per pixel — it is underdetermined.

## The aperture problem, made precise

Geometrically, the constraint fixes only the component of flow **along the image gradient** (i.e.
perpendicular to an edge). Write flow as a component along `∇I` plus a component along the edge:
the edge-parallel component multiplies `∇I·(edge direction) = 0` and vanishes from the equation.
So a small window sees only *normal flow* — the along-edge motion is invisible. This is the
**aperture problem**: through a narrow aperture you cannot tell how fast a straight edge slides. It
is not a numerical nuisance; it is a rank deficiency, made explicit next.

## Lucas-Kanade via the structure tensor

Lucas & Kanade (1981) assume flow is **constant over a small window** `W`. Stacking the constraint
over all pixels in `W` gives an overdetermined linear system `A·(u,v) = b`, solved by least squares.
The normal equations are `AᵀA·(u,v) = Aᵀb`, where `AᵀA` is the **structure tensor** (Week 2's corner
matrix!):

    M = Σ_W [ I_x²    I_x·I_y ]        b = -Σ_W [ I_x·I_t ]
            [ I_x·I_y  I_y²   ]                 [ I_y·I_t ]

Flow is recoverable iff `M` is invertible — i.e. both eigenvalues `λ₁ ≥ λ₂` are large. That is
exactly the **corner** condition. On a flat region both eigenvalues are ~0 (no texture, no signal);
on an edge one eigenvalue collapses (the aperture problem, now visible as a near-singular `M`). So
Lucas-Kanade works precisely where Harris corners fire, which is why classical trackers follow
corners.

```python
import cv2, numpy as np
# Sparse LK: track good corners across frames
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None)
# Dense Farneback: (H, W, 2) field of (u, v)
flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                     0.5, 3, 15, 3, 5, 1.2, 0)
```

## Horn-Schunck: the variational alternative

To get flow at **every** pixel — including the textureless ones LK abandons — Horn & Schunck (1981)
add a **global smoothness** prior and minimize an energy over the whole image:

    E(u,v) = ∫∫ (I_x·u + I_y·v + I_t)²  +  α²(‖∇u‖² + ‖∇v‖²)  dx dy

The first term is the data (brightness-constancy) cost; the second penalizes non-smooth flow, with
`α` trading data-fidelity for smoothness. Setting the variation to zero yields the **Euler-Lagrange
equations**, a pair of coupled PDEs whose discretization is the classic Jacobi/Gauss-Seidel update:

    u ← ū − I_x·(I_x·ū + I_y·v̄ + I_t) / (α² + I_x² + I_y²)      (v analogous, ū = local average)

Smoothness *propagates* motion from textured pixels into ambiguous ones — the prior fills the
aperture-problem gap. The cost is over-smoothing across true motion boundaries (occlusion edges),
which modern energies fix with robust (L1 / total-variation) penalties instead of the quadratic one.

## Large motion and coarse-to-fine

The Taylor expansion is only valid for **small** displacements. Fast motion breaks it. The classical
fix is a **coarse-to-fine image pyramid**: estimate flow on a downsampled image (where motion is
small in pixels), upsample it, warp the next frame toward the first, and refine the residual. This
warping idea reappears, learned, inside PWC-Net (Lecture 4).

## Visualizing flow

Flow is 2-D per pixel, shown as **color**: hue = motion direction (angle of `(u,v)`), and
saturation/value = magnitude `√(u²+v²)`. A person walking right glows one hue; the still background
stays dark. Reading these color-coded fields is how you *see* motion; the Middlebury and Sintel
benchmarks standardized this color wheel.

**Takeaway:** optical flow is the per-pixel motion field, derived from brightness constancy into a
single-equation-two-unknowns constraint — the aperture problem, which is a rank deficiency of the
structure tensor `M`. Close it locally (Lucas-Kanade, works where corners fire, `M` well-conditioned),
globally (Horn-Schunck's variational smoothness energy and its Euler-Lagrange PDEs), or with a
coarse-to-fine pyramid for large motion. Visualize as color; it is a foundational motion primitive,
not usually an end goal.
