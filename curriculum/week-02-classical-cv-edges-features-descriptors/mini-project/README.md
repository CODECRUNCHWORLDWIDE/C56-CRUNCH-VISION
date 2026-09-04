# Mini-Project — A Robust Feature-Matching Image Aligner

## Brief

Build a tool that aligns two views of the same scene using only classical features — the pipeline behind
panorama stitching, image registration, and visual localization — and *instrument it* so you can defend
every threshold and iteration count from this week's theory. This proves you understand features,
descriptors, matching, and robust geometry before neural features arrive.

## Requirements

1. **Detect & describe.** Extract ORB keypoints and descriptors in both images; add a SIFT path for
   comparison. Use the correct distance metric for each (Hamming for ORB, L2 for SIFT).
2. **Match & filter.** Brute-force `knnMatch(k=2)` + Lowe's ratio test *and* a mutual (cross-check)
   filter. Visualize surviving matches. Report counts after each filter stage.
3. **Robust geometry.** Estimate a homography with RANSAC. Report the inlier count vs. total good matches,
   *and* compute the RANSAC iteration budget your observed inlier ratio implies via
   `N ≥ log(1−p)/log(1−w^4)` — check it against the iterations OpenCV used.
4. **Align.** Warp one image into the other's frame and overlay/stitch them to show the alignment worked.
   Include a checkerboard or difference overlay to make residual misalignment visible.
5. **Instrument & report.** Report: keypoint counts, per-filter match counts, inlier count, chosen ρ and
   why, chosen RANSAC threshold τ and why (tie to expected keypoint localization noise), and one honest
   paragraph on where it fails (low overlap, repetitive texture, parallax, large viewpoint change).

## Stretch

- Extend to three or more images stitched into a wider panorama, chaining homographies to a common frame.
- Compare ORB vs. SIFT on the same pair for speed, match count, and inlier ratio; report a small table.
- Implement your own normalized DLT and RANSAC (from Exercise 4) and show it reproduces OpenCV's inliers.

## What you're proving

You can run the entire classical recognition-by-geometry pipeline *and reason quantitatively about it* —
choosing ρ, τ, and N from theory rather than trial and error, and diagnosing failures as "too few matches"
vs. "wrong geometric model." Next week a CNN starts *learning* features instead of you engineering them;
because you built and instrumented the classical version, you will know exactly what the network replaces
and what (the robust-geometry back end) it keeps.
