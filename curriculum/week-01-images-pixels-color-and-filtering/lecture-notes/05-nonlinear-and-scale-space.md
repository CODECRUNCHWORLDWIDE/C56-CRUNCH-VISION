# Lecture 5 — Beyond linear filters: edge-preserving filtering and scale space

Every filter in Lectures 3–4 was **linear**: output is a fixed weighted sum of inputs. Linearity
is powerful and analyzable, but it has a fatal weakness for images — it cannot tell a meaningful edge from
noise. A Gaussian blur removes noise *and* smears every edge, because it weights neighbors purely by spatial
distance, blind to whether they belong to the same object. This lecture covers the two ideas that fix this:
**nonlinear, edge-preserving filters** and **scale space** — and both are load-bearing for the rest of the
course (Forsyth & Ponce, *Computer Vision: A Modern Approach*, 2nd ed., 2011, Ch. 4–5).

## Why linear smoothing must blur edges

A linear low-pass filter attenuates high frequencies. But an edge *is* high-frequency content — a step in
intensity is a broad band of high frequencies. So any filter that kills the high-frequency noise necessarily
attenuates the edge. There is no linear escape: the trade-off between noise removal and edge preservation is
built into the frequency response. To preserve edges you must make the filter **content-adaptive** — i.e.
nonlinear.

## Median filter: order statistics beat averaging

The **median filter** replaces each pixel with the median of its neighborhood, not the mean. Because the
median is an order statistic, a few extreme outliers cannot pull it — so median filtering annihilates
**salt-and-pepper** (impulse) noise that a Gaussian merely spreads into gray smudges, while keeping edges
crisp (a step's median is still a step). It is nonlinear (median of a sum ≠ sum of medians), cannot be
written as a convolution, and is the standard first tool for impulse noise. Its cost is higher (a sort or
histogram per window) and it can erode fine detail and corners.

## Bilateral filter: distance in space *and* in intensity

The **bilateral filter** (Tomasi & Manduchi, 1998, "Bilateral Filtering for Gray and Color Images," ICCV)
generalizes the Gaussian by weighting neighbors on *two* axes — spatial closeness and **intensity
similarity**:

    BF[p] = (1/W_p) Σ_q  G_σs(‖p − q‖) · G_σr(|I_p − I_q|) · I_q,

where `G_σs` is the usual spatial Gaussian and `G_σr` a **range** Gaussian on the intensity difference. A
neighbor across a strong edge has a large intensity difference, so `G_σr` drives its weight to zero — the
filter averages *within* a region but not *across* its boundary. Result: noise is smoothed while edges stay
sharp. The price is that it is no longer shift-invariant or a convolution (the weights depend on the image
content), so it is slower and needs fast approximations (Paris & Durand, 2006). Bilateral filtering underlies
edge-aware denoising, tone mapping, and the "beauty" smoothing in phone cameras.

## Anisotropic diffusion: filtering as a PDE

A deeper view (Perona & Malik, 1990, "Scale-Space and Edge Detection Using Anisotropic Diffusion," IEEE
TPAMI) treats smoothing as **heat diffusion**: linear Gaussian blur is exactly the solution of the isotropic
heat equation `∂I/∂t = ∇·(∇I)` run for time `t = σ²/2`. Perona–Malik make the diffusion coefficient a
*decreasing function of gradient magnitude*, so diffusion halts at edges — smoothing flat regions while
freezing boundaries. This connects filtering to partial differential equations and is the ancestor of many
modern edge-preserving methods.

## Scale space: there is no single right blur

How much should you blur? The scale-space answer (Lindeberg, 1994, *Scale-Space Theory in Computer Vision*;
Witkin, 1983) is: **do not choose — represent the image at all scales at once.** Convolving with Gaussians of
increasing `σ` produces a one-parameter family `L(x, y; σ) = f * G_σ`, the **Gaussian scale space**. The
Gaussian is the *unique* kernel that generates a scale space without creating new structure as you coarsen
(no spurious extrema appear at larger scale) — a strong axiomatic result. Sampling this family and
subsampling (Lecture 4's anti-aliased downsample) gives the **Gaussian pyramid** (Burt & Adelson, 1983): the
same image at halving resolutions, the substrate for multi-scale detection, blending, and coarse-to-fine
search. Difference-of-Gaussians between adjacent pyramid levels approximates the Laplacian and is the blob
detector at the heart of SIFT (Week 2). So scale space is not a side topic — it is the bridge from this
week's filtering to next week's features.

**Takeaway:** linear filters cannot separate edges from noise because an edge *is* high frequency. Nonlinear
filters escape this: the median (an order statistic) kills impulse noise while keeping steps; the bilateral
filter weights neighbors by *both* spatial and intensity distance so it smooths within regions but not across
edges; anisotropic diffusion frames smoothing as an edge-stopping PDE. And because no single blur is
"right," Gaussian scale space represents the image at every `σ` at once — the pyramid that powers multi-scale
detection and directly feeds Week 2's feature detectors.
