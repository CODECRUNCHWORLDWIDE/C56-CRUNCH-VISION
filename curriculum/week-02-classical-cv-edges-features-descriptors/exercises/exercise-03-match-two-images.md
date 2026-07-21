# Exercise 3 — Match features between two views

**Goal:** the full detect → describe → match → filter pipeline.

## Tasks

1. Take two photos of the same scene from slightly different viewpoints (or download a stereo/matching pair).
2. Detect and describe ORB keypoints in both images.
3. Match with a brute-force Hamming matcher using `knnMatch(k=2)`, then apply Lowe's ratio test (0.75). Draw the surviving matches as lines between the two images.
4. Vary the ratio threshold (0.6, 0.75, 0.9) and show how it trades false matches for missed ones.

## Deliverable

A notebook drawing the good matches between the two images, plus a short note on how the ratio threshold trades precision for recall. If almost every match line is a wild diagonal, your ratio test is too loose or the images don't overlap — fix it.
