# Exercise 1 — Estimate and visualize optical flow

**Goal:** see motion as a dense field.

## Tasks

1. Take a short clip (or two consecutive frames with clear motion). Convert to grayscale.
2. Compute dense optical flow with `cv2.calcOpticalFlowFarneback` between consecutive frames.
3. Visualize the flow as a color image: map flow direction to hue and magnitude to brightness (HSV → BGR). A moving object should glow; static background should stay dark.
4. Also try sparse Lucas–Kanade flow tracking a few corner keypoints (Week 2) across frames and draw their motion trails.

## Deliverable

A notebook showing the color-coded dense flow field and the sparse feature trails on your clip, with a note on what the aperture problem means for flow along edges. You should be able to read motion from the colors.
