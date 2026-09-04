# Week 8 — Resources

Curated, free where possible. You do not need all of these — pick what fits how you learn.

- Bewley et al. (2016) *SORT* and Wojke et al. (2017) *Deep SORT* — the tracking-by-detection classics; read the papers, then the reference implementations.
- Cao et al. (2017) *OpenPose* and Newell et al. (2016) *Stacked Hourglass* — bottom-up (PAF) and top-down (heatmap) pose, the two canonical designs.
- Luiten et al. (2021) *HOTA* and the [TrackEval](https://github.com/JonathonLuiten/TrackEval) toolkit — the modern metric and a maintained evaluator for MOTA/IDF1/HOTA.
- Zhang et al. (2022) *ByteTrack* — near-SOTA MOT with a plain Kalman + IoU core and a two-pass low-score association; the highest-leverage simple upgrade.
- [torchvision keypoint & detection models](https://pytorch.org/vision/stable/models.html), [MMPose](https://github.com/open-mmlab/mmpose), and MediaPipe Pose docs — pretrained pose estimators to run in the exercises.
- Thrun, Burgard & Fox, *Probabilistic Robotics* (2005), Ch. 3 — the full Bayesian derivation of the Kalman filter as the optimal linear-Gaussian estimator.
- `scipy.optimize.linear_sum_assignment` and the `motmetrics` library — the Hungarian solver and a Python metrics package for MOTA/IDF1.
- First Principles of Computer Vision (YouTube) — tracking and the Kalman filter explained intuitively; a visual companion to Lecture 4.

> All external links are references, not endorsements. Prefer primary sources (papers, official docs) when they exist.
