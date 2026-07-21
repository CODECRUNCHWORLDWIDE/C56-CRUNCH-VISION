# Challenge 1 — Stitch a panorama

Turn feature matching into something real: stitch two overlapping photos into a panorama.

1. Take two overlapping photos (pan your phone slightly between shots). Detect, describe, and match features with the ratio test.
2. Use the good matches to estimate a **homography** with `cv2.findHomography(..., cv2.RANSAC)`. Explain what RANSAC is doing and why the ratio-test-filtered matches still need it.
3. Warp one image into the other's frame (`cv2.warpPerspective`) and blend them into a single wider image.
4. Report how many matches survived and how many RANSAC kept as inliers.

**Deliverable:** the stitched panorama plus a note on the match → homography → warp pipeline and RANSAC's role. When it fails (ghosting, misalignment), diagnose whether the problem is too few matches or a bad homography. This is exactly how phone panorama modes work.
