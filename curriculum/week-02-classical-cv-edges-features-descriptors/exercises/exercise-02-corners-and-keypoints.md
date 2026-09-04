# Exercise 2 — The structure tensor, Harris response, and repeatability

**Goal:** find repeatable features, and see the eigenvalue story of Lecture 2 with your own eyes.

## Tasks

1. **Structure tensor by hand.** Compute `I_x, I_y` (Sobel), then the three windowed sums
   `Σ I_x^2, Σ I_x I_y, Σ I_y^2` with a Gaussian window. For a handful of pixels (a flat patch, an edge,
   a corner), print the 2×2 matrix `M` and its two eigenvalues, and confirm they match the flat/edge/corner
   classification.
2. **Harris response.** Compute `R = det(M) - k·trace(M)^2` yourself (k = 0.04), threshold and apply
   spatial non-maximum suppression, and overlay the detected corners. Compare to `cv2.cornerHarris`.
3. **Scale is not built in.** Run your Harris detector on the image and on a 0.25× downscaled copy;
   show that many corners are lost — Harris has no scale invariance. Then run ORB (which does) on the same
   pair and compare.
4. **ORB keypoints with scale and orientation.** Detect ORB keypoints and draw them with rich flags
   (`cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`) so the circle sizes (scale) and angle lines
   (orientation) are visible.
5. **Repeatability test.** Rotate the image 30° and scale it 0.7×, detect keypoints again, warp the
   detected locations back with the known transform, and compute the *repeatability score* — the fraction
   of original keypoints re-detected within a few pixels. Report it for Harris vs. ORB.

## Deliverable

A notebook showing the printed structure-tensor eigenvalues for flat/edge/corner pixels, your Harris
response vs. OpenCV's, ORB keypoints with scale/orientation, and a numeric repeatability score comparing
Harris and ORB under rotation+scale. The lesson: a good feature is one you can re-find, and that requires
scale.
