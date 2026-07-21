# Mini-Project — A Feature-Matching Image Aligner

## Brief

Build a tool that aligns two views of the same scene using only classical features — the pipeline behind panorama stitching, image registration, and visual localization — proving you understand features, descriptors, and matching before neural features arrive.

## Requirements

1. **Detect & describe:** ORB (or SIFT) keypoints and descriptors in both images.
2. **Match & filter:** brute-force matching with `knnMatch(k=2)` and Lowe's ratio test; visualize the surviving matches.
3. **Geometry:** estimate a homography with RANSAC; report the inlier count vs. total good matches.
4. **Align:** warp one image into the other's frame and overlay or stitch them to show the alignment worked.
5. **Report:** keypoint counts, good-match counts, inlier counts, and one honest paragraph on where it fails (low overlap, repetitive texture, large viewpoint change).

## Stretch

- Extend to three or more images stitched into a wider panorama.
- Compare ORB vs. SIFT on the same pair for speed and match quality.

## What you're proving

You can run the entire classical recognition-by-geometry pipeline and reason about when it is the right tool. Next week a CNN starts *learning* features instead of you engineering them — and because you built the classical version, you'll know exactly what the network is replacing.
