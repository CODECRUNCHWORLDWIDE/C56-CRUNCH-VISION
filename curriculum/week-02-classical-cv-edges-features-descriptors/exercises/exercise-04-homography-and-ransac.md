# Exercise 4 — Normalized DLT and RANSAC from the inside

**Goal:** demystify `findHomography` — implement the DLT, add Hartley normalization, and drive RANSAC
by the iteration formula from Lecture 5.

## Part A — the DLT (paper + code)

1. From `x' ≃ H x`, derive the two linear equations per correspondence via the cross product
   `x' × (Hx) = 0`. Write the `2×9` block for one correspondence.
2. Implement the DLT: stack `n ≥ 4` correspondences into `A`, take the SVD, and read `h` as the smallest
   right singular vector. Reshape to `H`.
3. Verify against `cv2.getPerspectiveTransform` (exactly 4 points) on a synthetic quadrilateral.

## Part B — normalization matters

1. Run your DLT on raw pixel coordinates and on Hartley-normalized coordinates (translate to zero mean,
   scale to mean distance √2, then de-normalize). Compare the reprojection error on a synthetic pair with
   known `H`. Show the normalized version is dramatically more accurate/stable.

## Part C — RANSAC by the numbers

1. Corrupt a set of true correspondences with a known fraction of outliers (e.g. w = 0.6, 0.4).
2. Implement RANSAC over your DLT: minimal samples of 4, inlier test by symmetric transfer error < τ.
3. Set the iteration count from `N ≥ log(1−p)/log(1−w^s)` with p = 0.99, and confirm empirically that
   fewer iterations start to fail as w drops. Report recovered inlier count vs. ground truth.

## Deliverable

A notebook with your DLT matching OpenCV on clean data, a normalized-vs-unnormalized error comparison,
and a RANSAC that recovers the correct model at chosen (w, N) pairs — plus a one-line check that your
chosen N agrees with the formula. You will never treat `findHomography` as a black box again.
