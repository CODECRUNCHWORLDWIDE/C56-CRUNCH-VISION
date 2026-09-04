# Exercise 1 — Estimate and visualize optical flow (dense + sparse)

**Goal:** see motion as a dense field and connect it to the structure-tensor theory.

## Tasks

1. Take a short clip (or two consecutive frames with clear motion). Convert to grayscale.
2. Compute **dense** flow with `cv2.calcOpticalFlowFarneback` between consecutive frames.
3. Visualize the flow as a color image using the standard convention: map flow **direction to hue**
   and **magnitude to value** (build an HSV image, convert to BGR). A moving object should glow; the
   static background should stay dark. Overlay a color-wheel legend.
4. Also run **sparse** Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) tracking corner keypoints
   (`goodFeaturesToTrack`, Week 2) across several frames and draw their motion trails.
5. Pick one edge in the scene and one corner. For each, inspect the local 2x2 structure tensor
   `M = [[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]` (compute its eigenvalues). Explain which one LK can track and
   which suffers the aperture problem, using the eigenvalues.

## Deliverable

A notebook showing the color-coded dense flow field (with legend) and sparse feature trails on your
clip, plus the two structure-tensor eigenvalue readouts and a written explanation of why flow along an
edge is unrecoverable (aperture problem = near-singular `M`). Deliverable: you can read motion from
color *and* justify where flow is trustworthy from the eigenvalues.
