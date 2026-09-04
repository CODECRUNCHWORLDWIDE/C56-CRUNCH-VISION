# Week 2 — Reading List

Primary sources for Week 2. Start with the three starred essentials, then read for depth. These papers and chapters are where the ideas are stated correctly — prefer them to tutorials.

- **Canny (1986), 'A Computational Approach to Edge Detection,' *IEEE TPAMI* 8(6)** — the original detector derived from three optimality criteria; surprisingly readable and the source of the whole pipeline.
- **Lowe (2004), 'Distinctive Image Features from Scale-Invariant Keypoints,' *IJCV* 60(2)** — SIFT in full: DoG scale-space, sub-pixel refinement, edge rejection, orientation, the 128-D descriptor, and the ratio test.
- **Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd ed., 2004), Ch. 4** — homography estimation, the (normalized) DLT, and why normalization is mandatory; the standard reference for the geometry back end.
- Harris & Stephens (1988), 'A Combined Corner and Edge Detector,' *Alvey Vision Conference* — the structure tensor and the R = det − k·trace^2 response.
- Lindeberg (1998), 'Feature Detection with Automatic Scale Selection,' *IJCV* 30(2) — gamma-normalized derivatives and why scale becomes a detectable quantity; the theory under SIFT's scale invariance.
- Koenderink (1984), 'The Structure of Images,' *Biological Cybernetics* 50 — the causality argument that singles out the Gaussian as the unique scale-space kernel.
- Fischler & Bolles (1981), 'Random Sample Consensus,' *Communications of the ACM* 24(6) — RANSAC, robust model fitting from contaminated data; the iteration-count analysis.
- Rublee, Rabaud, Konolige & Bradski (2011), 'ORB: an efficient alternative to SIFT or SURF,' *ICCV* — Oriented FAST + rotated BRIEF, the fast free real-time default.
- Mikolajczyk & Schmid (2005), 'A Performance Evaluation of Local Descriptors,' *IEEE TPAMI* 27(10) — the benchmark defining repeatability/matching evaluation for descriptors.
- DeTone, Malisiewicz & Rabinovich (2018), 'SuperPoint: Self-Supervised Interest Point Detection and Description,' *CVPRW* — the learned successor that keeps the classical detect/describe/match skeleton.
- Sun, Shen, Wang, Bao & Zhou (2021), 'LoFTR: Detector-Free Local Feature Matching with Transformers,' *CVPR* — dense detector-free matching for low-texture / wide-baseline pairs where classical detectors fail.
- Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., 2022), Ch. 7 (features) & Ch. 8 (alignment/stitching) — the free, comprehensive textbook tying the whole week together.
