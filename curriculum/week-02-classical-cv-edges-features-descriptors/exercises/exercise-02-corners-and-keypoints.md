# Exercise 2 — Detect corners and keypoints

**Goal:** find repeatable features and see repeatability with your own eyes.

## Tasks

1. Run Harris corner detection on an image with clear corners (a checkerboard, a building). Overlay the detected corners.
2. Run ORB keypoint detection on the same image and draw the keypoints *with* their scale and orientation (`cv2.drawKeypoints` with rich flags). Note the circles' sizes and angle lines.
3. **Repeatability test:** rotate the image 30° and/or scale it, detect keypoints again, and visually check that many of the same physical points are still detected. Which detector re-finds points better?
4. Count keypoints found on the original vs. a heavily blurred version and explain the drop.

## Deliverable

A notebook with Harris corners, ORB keypoints (showing scale/orientation), and the rotate/scale repeatability comparison. The lesson is that a good feature is one you can re-find.
