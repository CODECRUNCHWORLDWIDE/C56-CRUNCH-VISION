# Week 2 — Graduate Problem Set: Edges, the Structure Tensor, Scale-Space, and Robust Geometry

Twelve problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution
sketches follow — attempt each fully first. Notation: `I(x,y)` is image intensity, `∇I = (I_x, I_y)` its
gradient, `G_σ` a Gaussian of standard deviation σ, `M` the second-moment matrix, `H` a homography.

**P1 (edges as derivatives).** Show that convolving an image with the first derivative of a Gaussian,
`(dG_σ/dx) * I`, equals differentiating the smoothed image, `d/dx (G_σ * I)`. Which property of convolution
makes this hold, and why does it justify Canny's "blur then differentiate" order?

**P2 (NMS on a step).** A 1-D noiseless step edge produces the gradient profile `[0, 5, 5, 0]` at four
consecutive pixels. Explain why naive thresholding reports a two-pixel-wide edge and how non-maximum
suppression yields a single response. What tie-breaking issue arises and how is it handled?

**P3 (structure tensor derivation).** Starting from the windowed SSD
`E(u,v) = Σ w(x,y)[I(x+u,y+v) − I(x,y)]^2`, use a first-order Taylor expansion to derive
`E(u,v) ≈ [u v] M [u v]^T` and give `M` explicitly. Prove `M` is symmetric positive semi-definite.

**P4 (Harris response cases).** Given `M` with eigenvalues `(λ1, λ2)`, compute `R = det(M) − k·trace(M)^2`
for (a) `(0,0)`, (b) `(100, 0.1)`, (c) `(80, 60)`, with `k = 0.04`. Classify each pixel and state the sign
of `R` in each regime.

**P5 (DoG approximates LoG).** From the heat equation `∂G_σ/∂σ = σ ∇^2 G_σ` (using the σ-parameterization),
derive `G_{kσ} − G_σ ≈ (k−1) σ^2 ∇^2 G_σ`. Explain why this makes the Difference-of-Gaussian a
scale-normalized Laplacian and why the `σ^2` factor is exactly the normalization scale selection needs.

**P6 (scale selection).** For an idealized bright circular blob of radius `r` on a dark background, the
scale-normalized Laplacian response over scale is extremized at `t = r^2/2` (equivalently `σ = r/√2`).
Sketch the argument and explain, in one sentence, why the peak location *reads off* the blob's size.

**P7 (descriptor invariance).** Explain precisely which construction step of the SIFT descriptor provides
each invariance: (a) rotation, (b) uniform (multiplicative) illumination change, (c) small spatial shift,
(d) large non-linear illumination change (saturation). Name the step for each.

**P8 (ratio test probability).** Model the nearest and second-nearest descriptor distances of a *false*
match as roughly equal, and of a *true* match as clearly separated. Argue qualitatively why a threshold on
`d1/d2 < ρ` separates the two populations, and what happens to precision and recall as ρ → 1 and ρ → 0.

**P9 (homography DoF and DLT).** Explain why a 3×3 homography has 8, not 9, degrees of freedom. From
`x' × (Hx) = 0`, show each correspondence contributes 2 independent linear equations, and conclude the
minimal number of correspondences. Why is the DLT solution the smallest right singular vector of `A`?

**P10 (Hartley normalization).** Explain why running the DLT on raw pixel coordinates (values in the
hundreds) is ill-conditioned. State the Hartley normalization (zero mean, mean distance √2) and show how the
final homography is recovered as `H = T'^{-1} H̃ T`. Why does conditioning matter for the *smallest*
singular vector specifically?

**P11 (RANSAC iterations).** Derive `N ≥ log(1−p)/log(1−w^s)` from the requirement that at least one of `N`
minimal samples (size `s`, inlier ratio `w`) is all-inliers with probability ≥ p. Evaluate `N` for
`(s, w, p) = (4, 0.5, 0.99)` and `(4, 0.3, 0.99)`. Explain why lower-DoF models are cheaper to fit robustly.

**P12 (open analysis).** Classical detectors (SIFT/ORB) are similarity-covariant; real viewpoint change is
affine/projective, and low-texture scenes defeat keypoint detection entirely. Argue where the classical
detect→describe→match pipeline must break, why detector-free learned matchers (LoFTR) help in those regimes,
and — crucially — what part of the pipeline learning does *not* replace. Be specific about the
invariance hierarchy translation ⊂ similarity ⊂ affine ⊂ projective. (Open-ended; argue carefully.)

---

## Solution sketches

**S1.** Differentiation is a convolution (with `dδ/dx`), and convolution is associative and commutative:
`(dG/dx) * I = d/dx(G) * I = d/dx(G * I)`. So you may pre-compute the derivative-of-Gaussian kernel once;
blur-then-differentiate and differentiate-the-blur are identical — justifying Canny's staged order and its
efficiency.
**S2.** Thresholding keeps both pixels of the equal-height ridge (two-pixel edge). NMS compares each pixel
to its neighbours along the gradient and keeps only a local max; the tie is broken by sub-pixel parabolic
interpolation or a consistent rule, leaving one pixel — the single-response criterion.
**S3.** Taylor: `I(x+u,y+v)−I ≈ I_x u + I_y v`, so `E ≈ Σ w (I_x u + I_y v)^2 = [u v] M [u v]^T` with
`M = Σ w [[I_x^2, I_xI_y],[I_xI_y, I_y^2]]`. Symmetric by construction; for any vector `z`,
`z^T M z = Σ w (∇I·z)^2 ≥ 0` (w ≥ 0), so PSD.
**S4.** (a) `R = 0` — flat. (b) `det = 10`, `trace = 100.1`, `R ≈ 10 − 0.04·10020 ≈ −391` (< 0) — edge.
(c) `det = 4800`, `trace = 140`, `R = 4800 − 0.04·19600 = 4016` (> 0, large) — corner.
**S5.** Finite-difference the σ-heat equation: `G_{kσ} − G_σ ≈ (kσ−σ)·∂G/∂σ = (k−1)σ·(σ∇^2 G) =
(k−1)σ^2 ∇^2 G`. The `σ^2` is exactly the γ=1 scale-normalization that makes the Laplacian's extremum
scale-covariant, so DoG extrema select scale.
**S6.** The normalized LoG of a blob is a smooth function of `t` with a single interior extremum; matching
the Gaussian's effective radius to the blob radius maximizes the (signed) response at `t = r^2/2`. The
extremum's `t` therefore encodes `r`.
**S7.** (a) rotation: rotating the patch/histogram bins to the canonical orientation. (b) multiplicative
illumination: L2-normalizing the descriptor. (c) spatial shift: the spatial histogram binning (4×4 cells)
tolerates small shifts. (d) saturation/non-linear: clamping bin values at 0.2 then renormalizing.
**S8.** For a false match, `d1 ≈ d2` so `d1/d2 ≈ 1` and it fails `< ρ`; for a true match `d1 ≪ d2` so the
ratio is small and it passes. ρ → 1 admits almost everything (high recall, low precision); ρ → 0 admits
only near-perfect matches (high precision, low recall).
**S9.** `H` scale-invariant ⇒ 9 − 1 = 8 DoF. The cross product gives 3 equations, only 2 independent (the
third is a linear combination), so 2 per correspondence; 8 DoF / 2 = 4 correspondences. `Ah = 0` is
homogeneous; the least-squares unit-norm solution is the right singular vector with smallest singular value.
**S10.** Entries of `A` mix terms of order 1 and order `x·x' ~ 10^4–10^5`, so columns span many orders of
magnitude and the smallest singular vector is swamped by numerical error. Normalizing to zero mean and mean
distance √2 conditions `A`; de-normalize via `H = T'^{-1} H̃ T`. The *smallest* singular vector is most
sensitive to conditioning, so normalization is decisive.
**S11.** P(sample all-inlier) = `w^s`; P(all N contaminated) = `(1−w^s)^N ≤ 1−p` ⇒
`N ≥ log(1−p)/log(1−w^s)`. `(4,0.5,0.99) ⇒ N ≈ 72`; `(4,0.3,0.99) ⇒ N ≈ 566`. Lower `s` makes `w^s`
larger, so far fewer samples are needed — cheap robustness.
**S12.** Detectors fail where they find no repeatable keypoints (textureless/day-night) or where similarity
invariance is insufficient (large affine/projective viewpoint change). Detector-free matchers (LoFTR)
establish dense correspondences directly via attention, bypassing detection. What learning does *not*
replace: the robust geometry (RANSAC + DLT / essential-matrix), which still consumes the correspondences —
the boundary between the learned appearance front end and the classical geometry back end.
