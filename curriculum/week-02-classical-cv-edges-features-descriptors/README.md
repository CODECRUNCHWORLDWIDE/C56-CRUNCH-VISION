# Week 2 — Classical CV: edges, features, descriptors

> **Goal:** by Sunday you can (1) build a Canny detector and justify each stage from Canny's own optimal-detection criteria rather than as a recipe, (2) derive the Harris response from the second-moment matrix and read its eigenvalues, and explain scale and orientation assignment from scale-space theory, (3) describe what SIFT and ORB actually encode and why the ratio test and mutual-nearest-neighbour filtering work, and (4) estimate a homography with the normalized DLT and reject outliers with RANSAC, predicting the iteration count from the outlier rate — the full detect -> describe -> match -> fit pipeline that still ships in phones, drones, and SLAM, with the theory that tells you when it will fail.

Before deep learning, computer vision *was* hand-engineered features — and a surprising amount of it still runs in production: your phone's panorama and night-mode alignment, visual-inertial SLAM on drones and headsets, image registration in medical and satellite pipelines, and the front end of structure-from-motion systems like COLMAP. This week you build the classical pipeline properly: **edges** (Canny), **corners** (Harris), **scale-invariant keypoints and descriptors** (SIFT/ORB), **matching**, and the **robust geometry** (homography + RANSAC) that turns noisy correspondences into a usable model.

The usual treatment presents each step as a function call with a recipe attached. We will not. You will see that Canny is not a heuristic but the solution to a stated optimization problem — good detection, good localization, and a single response per edge (Canny, 1986, *IEEE TPAMI*). You will derive the Harris corner response from the auto-correlation of a windowed patch and read cornerness directly off the eigenvalues of the second-moment matrix (Harris & Stephens, 1988). You will meet scale-space theory — why a Gaussian is the *unique* smoothing kernel that creates no new structure (Koenderink, 1984; Lindeberg, 1998) — and how SIFT exploits it for scale invariance (Lowe, 2004, *IJCV*). Finally you will treat matching as robust estimation: the normalized Direct Linear Transform for homographies (Hartley & Zisserman, 2004) and RANSAC (Fischler & Bolles, 1981), whose iteration budget you can compute in closed form from the inlier ratio.

By the end you will not merely call `cv2.Canny` and `cv2.findHomography` — you will know what each is solving, when it breaks, and what the learned successors (SuperPoint, LoFTR) replace and what they keep.

## Learning objectives

By the end of this week, you will be able to:

- **Derive** the Canny detector from its three optimality criteria (detection, localization, single response) and explain non-maximum suppression and hysteresis as the discrete realization of that objective.
- **Construct** the second-moment (structure) tensor from image gradients and classify a pixel as flat, edge, or corner from its two eigenvalues, deriving the Harris response `R = det(M) - k*trace(M)^2`.
- **Explain** scale-space theory: why the Gaussian is the unique scale-space kernel, and how the Laplacian-of-Gaussian and its Difference-of-Gaussian approximation enable automatic scale selection (Lindeberg's gamma-normalization).
- **Characterize** what SIFT (128-D gradient-orientation histograms) and ORB (256-bit BRIEF) descriptors encode, and justify rotation and illumination invariance from their construction.
- **Match** descriptors with nearest-neighbour search, and defend Lowe's ratio test and mutual-consistency checks in terms of match ambiguity and precision/recall.
- **Estimate** a homography with the normalized Direct Linear Transform, explaining why Hartley normalization is essential for conditioning, and count the degrees of freedom.
- **Predict** RANSAC's required iteration count from the inlier ratio and sample size, and explain the inlier-threshold / model-complexity trade-offs that make it robust.
- **Judge** when classical features beat learned ones — training-free, real-time-on-CPU, interpretable, geometry-exact — and name the learned successors and what they change.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 4670` — detect edges, corners and scale-invariant interest points, describe them with local descriptors, match them across views, and estimate the transformation between two views robustly. |
| Industry | Align two photographs of the same scene and report how many correspondences actually survived the robust fit, not how many were proposed. |
| Beyond the bar | predicts the RANSAC iteration count from the inlier ratio and sample size, then makes the learner check a real run against that prediction — `exercises/exercise-04-homography-and-ransac.md` |

## Prerequisites

- Week 1's mini-project (image loading, grayscale conversion, Gaussian blur, Sobel gradients), committed and working.
- Linear algebra: matrix multiplication, eigenvalues/eigenvectors of a symmetric 2x2 matrix, the SVD (at least conceptually).
- Calculus: partial derivatives, the gradient, the Laplacian, Taylor expansion to second order.
- Basic probability: independence, and the probability that k independent trials all succeed (for the RANSAC count).

## This week

**Lectures**

1. [Lecture 1 — Edges and the Canny pipeline as optimal detection](lecture-notes/01-edges-and-canny.md)
2. [Lecture 2 — Corners, the structure tensor, and scale-space keypoints](lecture-notes/02-corners-and-keypoints.md)
3. [Lecture 3 — Descriptors, matching, and robust correspondence](lecture-notes/03-descriptors-and-matching.md)
4. [Lecture 4 — Scale-space theory, the Laplacian of Gaussian, and SIFT internals](lecture-notes/04-scale-space-and-sift.md)
5. [Lecture 5 — Robust two-view geometry, RANSAC, and the learned successors](lecture-notes/05-robust-geometry-and-learned-features.md)

**Exercises**

1. [Exercise 1 — Build, tune, and justify a Canny edge detector](exercises/exercise-01-canny-pipeline.md)
2. [Exercise 2 — The structure tensor, Harris response, and repeatability](exercises/exercise-02-corners-and-keypoints.md)
3. [Exercise 3 — Match features and quantify precision/recall of the filters](exercises/exercise-03-match-two-images.md)
4. [Exercise 4 — Normalized DLT and RANSAC from the inside](exercises/exercise-04-homography-and-ransac.md)

**Challenges**

1. [Challenge 1 — Stitch a multi-image panorama and diagnose its failures](challenges/challenge-01-panorama-stitch.md)
2. [Challenge 2 — When would you *not* use deep learning? Benchmark it.](challenges/challenge-02-classical-vs-deep.md)
3. [Challenge 3 — Push invariance: affine-covariant regions and viewpoint change](challenges/challenge-03-affine-covariant-detector.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 3.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook that takes two photos of the same scene from different viewpoints, detects and describes keypoints in each (ORB and, for comparison, SIFT), matches them with a ratio test plus mutual-consistency filter, estimates a homography with RANSAC (reporting inliers vs. good matches), and draws the surviving correspondences as lines between the images — with a short written justification of the ratio threshold and RANSAC iteration count chosen.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
