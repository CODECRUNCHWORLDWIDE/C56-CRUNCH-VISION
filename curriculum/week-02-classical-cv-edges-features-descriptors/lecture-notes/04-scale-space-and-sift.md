# Lecture 4 — Scale-space theory, the Laplacian of Gaussian, and SIFT internals

Rotation invariance is easy — rotate the patch to a canonical orientation. **Scale** invariance is
the hard, beautiful part of classical features, and it rests on a genuinely deep result: among all
smoothing kernels, the Gaussian is the *unique* one that defines a well-behaved scale-space. This lecture
derives why, then shows how SIFT turns the theory into a detector.

## What a scale-space is, and why the Gaussian is forced

A **scale-space** embeds an image `I(x, y)` into a one-parameter family `L(x, y, t)` of increasingly
coarse versions, with `t ≥ 0` the scale. We demand it create no spurious structure as scale increases —
coarsening must only *merge or remove* detail, never invent it. Formalizing "no new structure" as a set
of axioms (linearity, shift- and rotation-invariance, and *causality*: no new level sets / local extrema
are created as `t` grows) forces a unique answer: the family must satisfy the **heat (diffusion)
equation** `∂L/∂t = (1/2) ∇^2 L`, whose Green's function is the **Gaussian**. So

    L(x, y, t) = G_{√t}(x, y) * I(x, y),     G_σ(x,y) = (1/2πσ^2) exp(-(x^2+y^2)/2σ^2).

This is the Koenderink (1984, "The structure of images," *Biological Cybernetics*) / Witkin (1983) /
Lindeberg (1994, *Scale-Space Theory in Computer Vision*) result: the Gaussian is not one option among
many — it is the *only* linear kernel that guarantees causality in scale. Every scale-invariant classical
detector builds on Gaussian scale-space for this reason.

## Automatic scale selection: gamma-normalized derivatives

Blurring alone does not tell you the *right* scale of a structure. Lindeberg (1998, "Feature detection
with automatic scale selection," *IJCV*) showed that if you form **scale-normalized** derivative
operators and look for extrema *over scale as well as space*, the scale at which a normalized response
peaks matches the characteristic size of the structure. The workhorse is the scale-normalized Laplacian:

    ∇^2_norm L = t (L_xx + L_yy).

For a blob of radius `r`, `∇^2_norm L` attains its extremum at scale `t = r^2 / 2` — i.e. the response
peak *reads off the blob size*. This is automatic scale selection: search `(x, y, t)` for extrema of the
normalized Laplacian, and each extremum gives both a location and a scale. Without the `t` normalization
the Laplacian's amplitude decays with scale and no clean extremum exists — the normalization is what makes
scale a detectable quantity.

## Difference-of-Gaussian: the cheap Laplacian

The Laplacian-of-Gaussian (LoG) is expensive to compute at every scale. Two Gaussians at nearby scales
differ by (approximately) a scaled Laplacian — this follows from the heat equation `∂G/∂t = (1/2)∇^2 G`,
approximated by a finite difference in scale:

    G_{kσ} - G_σ ≈ (k - 1) σ^2 ∇^2 G_σ.

So the **Difference-of-Gaussian (DoG)** `D = (G_{kσ} - G_σ) * I = L(·, kσ) - L(·, σ)` is a near-free
approximation of the scale-normalized LoG, obtained by subtracting adjacent pyramid levels you already
computed while building the Gaussian pyramid. SIFT detects keypoints as **local extrema of `D` over the
3×3×3 neighbourhood in space and scale** — each candidate is larger (or smaller) than all 26 neighbours.

## The SIFT pipeline, precisely (Lowe, 2004, *IJCV* 60(2))

1. **Build the Gaussian pyramid** across octaves (each octave halves resolution) and `s` intra-octave
   scales; subtract adjacent scales to form the DoG pyramid.
2. **Detect extrema** of DoG over space and scale (26-neighbour test).
3. **Sub-pixel refine** each extremum by fitting a 3-D quadratic (second-order Taylor of `D`) and solving
   for the offset; this also gives a refined scale and the extremum value.
4. **Reject weak and edge responses.** Discard low-contrast extrema (`|D| < threshold`). Discard edge-like
   responses using the *Hessian of D*: the ratio of principal curvatures `Tr(H)^2 / det(H) < (r+1)^2/r`
   (Lowe uses `r = 10`) — exactly the Harris idea, reused to reject points that are well-localized in only
   one direction.
5. **Assign orientation** from a 36-bin gradient-orientation histogram over the keypoint neighbourhood
   (peaks within 80% of the max spawn multiple keypoints), giving rotation invariance.
6. **Build the descriptor** — the 4×4×8 = 128-D gradient-histogram vector of Lecture 3, computed in the
   orientation-normalized, scale-normalized frame.

## Worked reasoning: why DoG finds blobs, not edges

At a bright blob on a dark background, `L` has a local minimum of curvature matched at scale `t ≈ r^2/2`,
so `D` (approximating `t∇^2 L`) shows a clean extremum — isolated in both space and scale. At an edge, the
curvature is large across the edge but ~zero along it, so the DoG response is a ridge, not an isolated
extremum, and step 4's principal-curvature ratio (`> (r+1)^2/r`) rejects it. This is why SIFT keypoints
are "blobs / corners" and not edge pixels: the detector's own second-derivative structure encodes the
two-direction-change requirement from Lecture 2.

## Common pitfalls

- **Too few intra-octave scales.** With too coarse a scale sampling, the DoG extrema are poorly
  localized in scale and repeatability drops; Lowe found `s = 3` scales per octave near-optimal.
- **Forgetting the contrast/edge rejection.** Without steps 4's two filters, SIFT floods low-contrast and
  edge regions with unstable keypoints that destroy matching.
- **Confusing DoG with a generic band-pass.** DoG works *because* it approximates the scale-normalized
  Laplacian from the heat equation; the `(k-1)σ^2` factor and adjacent-scale subtraction are what make it
  a scale-selection operator, not just an edge filter.

**Takeaway:** the Gaussian is the unique causal scale-space kernel (heat equation / Koenderink–Lindeberg),
and scale-normalized derivatives make *scale* a detectable quantity whose extrema read off structure size
(Lindeberg's automatic scale selection). SIFT realizes this cheaply with Difference-of-Gaussian extrema,
refines and filters them with sub-pixel Taylor fits and a Harris-style principal-curvature test, then
attaches orientation and a 128-D histogram descriptor — turning deep scale-space theory into a fast,
repeatable, scale-invariant detector.
