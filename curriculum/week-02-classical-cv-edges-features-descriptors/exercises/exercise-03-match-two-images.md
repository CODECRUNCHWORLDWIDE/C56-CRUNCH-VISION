# Exercise 3 — Match features and quantify precision/recall of the filters

**Goal:** the full detect → describe → match → filter pipeline, measured rather than eyeballed.

## Tasks

1. Take two photos of the same scene from slightly different viewpoints (or download a matching pair, e.g.
   from the Oxford/HPatches sets). Convert to grayscale.
2. Detect and describe ORB keypoints in both images; separately do the same with SIFT
   (`cv2.SIFT_create`). Note the descriptor dtype and the correct distance metric for each (Hamming vs. L2).
3. Match with `knnMatch(k=2)`, apply Lowe's ratio test, and *also* apply a mutual (cross-check) filter.
   Draw the surviving matches as lines between the two images.
4. **Sweep the ratio.** For ρ ∈ {0.6, 0.7, 0.75, 0.8, 0.9}, count surviving matches and (using a RANSAC
   homography as an approximate ground-truth inlier oracle) estimate precision (fraction of survivors that
   are RANSAC inliers) and recall (inliers kept vs. total inliers at ρ=0.9). Plot precision and recall vs. ρ.
5. Repeat the whole pipeline for SIFT vs. ORB and compare match counts, precision, and wall-clock time.

## Deliverable

A notebook drawing the good matches, a precision/recall-vs-ρ plot, and an ORB-vs-SIFT comparison table
(matches, precision, time). A short note: how does ρ trade precision for recall, and where would you set
it for a stitch vs. for a loop-closure detector? If almost every match line is a wild diagonal, your ratio
is too loose or the images do not overlap — fix it.
