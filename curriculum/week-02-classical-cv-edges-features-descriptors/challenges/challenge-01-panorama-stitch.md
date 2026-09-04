# Challenge 1 — Stitch a multi-image panorama and diagnose its failures

Turn feature matching into something real: stitch overlapping photos into a panorama, then push it
until it breaks and explain why.

1. Take three or more overlapping photos (pan your phone, keeping roughly one center of rotation). Detect,
   describe, and match features pairwise with the ratio + mutual tests.
2. Estimate each pairwise **homography** with `cv2.findHomography(..., cv2.RANSAC)`. Report, per pair, the
   good-match count and the RANSAC inlier count, and compute the RANSAC iteration budget your inlier ratio
   implies via `N ≥ log(1−p)/log(1−w^4)`.
3. Chain the homographies to a common reference frame, warp each image (`cv2.warpPerspective`) into a
   canvas, and blend the overlaps (feathering or multi-band blending) into one wide image.
4. **Break it on purpose.** Introduce parallax by translating the camera between shots (not pure rotation)
   and photograph a non-planar scene. Show the ghosting/misalignment and explain *why* a single homography
   cannot model it — tie it to the planar/pure-rotation assumption of Lecture 5.

**Deliverable:** the stitched panorama, a per-pair table (matches / inliers / implied N), and a written
failure analysis distinguishing "too few matches" from "wrong geometric model (parallax)." This is exactly
how phone panorama modes work — and exactly how they fail.
