# Week 2 — Homework

Lock in the classical toolkit — and its theory — before CNNs start learning features for you. These take a few focused hours; do the derivations by hand before coding, so the structure tensor, scale-space, and the RANSAC count are muscle memory before a library hides them.

## Tasks

- Re-derive, on paper without notes, the second-moment matrix M from the windowed SSD via first-order Taylor expansion, and state the eigenvalue conditions for flat / edge / corner.
- Explain, in your own words, what each of Canny's five stages fixes about naive gradient thresholding, mapping each stage to one of Canny's three optimality criteria.
- Derive that Difference-of-Gaussian approximates the scale-normalized Laplacian-of-Gaussian, and explain in two sentences why this gives SIFT scale selection.
- Add RANSAC-based homography estimation to your matching notebook; report inliers vs. good matches and compute the RANSAC iteration budget your inlier ratio implies via N ≥ log(1−p)/log(1−w^4).
- Read the Hartley & Zisserman treatment (or the OpenCV docs) and write a short note on why Hartley normalization is required before the DLT, with a worked conditioning example.
- Read the SIFT and ORB abstracts and summarize when to prefer each (accuracy vs. speed, real-valued vs. binary, licensing) in a decision table with the correct distance metric for each.

## Definition of done

A committed notebook that, given two overlapping images, detects and describes ORB (and SIFT) keypoints, matches them with a ratio + mutual filter, estimates a RANSAC homography, aligns/overlays or stitches the images, and reports keypoint counts, per-filter match counts, inlier counts, and the chosen ρ/τ/N with a written justification tying each to this week's theory — plus an honest failure paragraph distinguishing too-few-matches from wrong-model (parallax).

Submit by committing your work to your course repo under `week-02/`.
