# Lecture 5 — Robust two-view geometry, RANSAC, and the learned successors

Matched keypoints are noisy and contaminated with outliers. Turning them into a usable geometric
model — the homography behind a panorama, the essential matrix behind SLAM — is a problem of **robust
estimation**: fit a model that a majority of correspondences agree on, while ignoring a minority that
disagree arbitrarily. This lecture derives the two tools that do it (the normalized DLT and RANSAC) with
enough rigor to predict cost and failure, then surveys what deep learning has changed.

## The homography and its degrees of freedom

For two images of a *planar* scene (or a scene viewed under *pure camera rotation*), corresponding image
points `x = (x, y, 1)^T` and `x' = (x', y', 1)^T` (homogeneous coordinates) are related by a `3×3`
**homography** `H`: `x' ≃ H x`, where `≃` means "equal up to scale." `H` has 9 entries but is defined
only up to scale, so it has **8 degrees of freedom**. Each point correspondence gives 2 independent
equations (x and y), so a minimum of **4 correspondences** (no three collinear) determines `H` — the
minimal sample RANSAC will use.

## The Direct Linear Transform, and why normalization is mandatory

From `x' ≃ H x`, the cross product `x' × (H x) = 0` eliminates the unknown scale and yields, per
correspondence, two linear equations in the entries of `H`. Stacking `n ≥ 4` correspondences gives a
homogeneous system `A h = 0` where `h` is the 9-vector of `H`'s entries. The solution is the right
singular vector of `A` with the smallest singular value (the SVD gives it directly) — this is the
**Direct Linear Transform (DLT)**.

Naively, DLT is badly conditioned: pixel coordinates in the hundreds make `A`'s columns span wildly
different magnitudes (some entries scale like `x·x'`, i.e. ~10^4–10^5), so the smallest singular vector is
dominated by numerical noise. **Hartley normalization** (Hartley, 1997, "In defense of the eight-point
algorithm," *IEEE TPAMI*; Hartley & Zisserman, *Multiple View Geometry*, 2004, §4.4) fixes this: translate
each image's points to zero mean and scale so their mean distance from the origin is √2, run the DLT on the
normalized points to get `H̃`, then de-normalize `H = T'^{-1} H̃ T`. This single pre-conditioning step is
the difference between a usable and a useless estimate; it is not optional.

## RANSAC, and its iteration budget in closed form

Even normalized least-squares is not robust — one gross outlier corrupts every entry. **RANSAC**
(Fischler & Bolles, 1981, "Random Sample Consensus," *Comm. ACM*) instead:

1. Sample a **minimal set** (here `s = 4` correspondences) at random.
2. Fit the model (DLT homography) from that set.
3. **Score** by counting inliers: correspondences whose reprojection error `‖x' − H x‖` (symmetric
   transfer error is better) is below a threshold `τ`.
4. Repeat `N` times; keep the model with the most inliers, then refit on all its inliers.

How many iterations `N`? Let `w` be the inlier ratio (fraction of correct matches). The probability that a
single minimal sample of size `s` is all-inliers is `w^s`; that it is contaminated is `1 − w^s`; that
*all* `N` samples are contaminated is `(1 − w^s)^N`. To succeed (at least one clean sample) with
probability `p`, require `(1 − w^s)^N ≤ 1 − p`, so

    N ≥ log(1 − p) / log(1 − w^s).

Worked example: `s = 4`, inlier ratio `w = 0.5`, target `p = 0.99`. Then `w^s = 0.0625`, and
`N ≥ log(0.01)/log(0.9375) ≈ 4.605/0.0645 ≈ 72` iterations. At `w = 0.3`, the same computation gives
`N ≈ 566` — the budget explodes as outliers rise and as the minimal-sample size `s` grows, which is why
low-DoF models (fewer points per sample) are dramatically cheaper to fit robustly. This formula is the
knob you tune: set `N` from your worst-case expected `w`.

```python
import cv2, numpy as np
src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
inliers = int(mask.sum())   # report inliers vs. len(good)
```

Modern variants improve on vanilla RANSAC: **MLESAC** scores by likelihood not raw counts; **PROSAC**
samples high-confidence matches first; **LO-RANSAC** adds a local optimization step; **MAGSAC++**
(Barath et al., 2020, *CVPR*) removes the hard threshold `τ` by marginalizing over it — now a common
default.

## The reprojection-threshold / model trade-off

The inlier threshold `τ` trades precision for recall: too tight and correct matches with normal
localization noise are discarded (few inliers, unstable fit); too loose and outliers sneak in (biased
model). A principled `τ` scales with the expected keypoint localization noise (a few pixels). Likewise,
choosing the *model* matters: fit a homography only when the scene is planar or the motion is pure
rotation; otherwise parallax makes no single `H` valid and you need the fundamental/essential matrix
(7/5-DoF, an 8- or 5-point algorithm) — using the wrong model guarantees failure no matter how good the
matches.

## What learned features change — and keep

Since ~2018 learned local features have overtaken SIFT on hard (wide-baseline, low-texture, day/night)
matching, while keeping the *same* detect → describe → match → robust-fit skeleton:

- **SuperPoint** (DeTone et al., 2018, *CVPRW*) — a single CNN producing keypoints and descriptors jointly,
  trained self-supervised via "homographic adaptation."
- **D2-Net** (Dusmanu et al., 2019) and **R2D2** (Revaud et al., 2019) — describe-then-detect and jointly
  learned reliability/repeatability.
- **SuperGlue** (Sarlin et al., 2020, *CVPR*) — a graph-neural-network *matcher* that reasons about all
  correspondences jointly with attention, replacing the ratio test with learned global assignment.
- **LoFTR** (Sun et al., 2021, *CVPR*) — *detector-free* dense matching via transformers, excelling in
  low-texture scenes where keypoint detectors find nothing.

What they keep is instructive: correspondences still feed **RANSAC + DLT** (or PnP / essential-matrix)
geometry — the robust-estimation layer of this lecture is *unchanged*. Learning replaced the
hand-engineered *appearance* front end; the *geometry* back end is still the classical algebra you derived
here. That is the honest boundary between the two eras.

## Common pitfalls

- **Skipping Hartley normalization.** Unnormalized DLT is numerically hopeless on pixel coordinates.
- **Fixing `N` without regard to `w`.** If the true inlier ratio is lower than assumed, RANSAC silently
  returns a garbage model; compute `N` from a conservative `w`.
- **Forcing a homography on a non-planar scene.** Parallax breaks the single-`H` assumption; ghosting in a
  stitch is the symptom. Diagnose "too few matches" vs. "wrong model" separately.

**Takeaway:** turning matches into geometry is robust estimation. A homography has 8 DoF and needs 4
points; estimate it with the *normalized* DLT (Hartley normalization is mandatory), and reject outliers
with RANSAC, whose iteration budget `N ≥ log(1−p)/log(1−w^s)` you can compute from the inlier ratio.
Learned features (SuperPoint, SuperGlue, LoFTR) have replaced the hand-crafted appearance front end on
hard problems — but they still feed the *same* classical robust-geometry back end you derived this week.
