# Week 9 — Resources

Curated, free where possible. You do not need all of these — pick what fits how you learn.

- [OpenCV — Optical Flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) — Lucas-Kanade (sparse) and Farneback (dense) in runnable code; the fastest way to reproduce Lecture 1.
- [torchvision — video classification models & the Kinetics dataset](https://pytorch.org/vision/stable/models.html#video-classification) — R(2+1)D, MC3, and the exact preprocessing/weights for Exercise 2 and the mini-project.
- [torchvision — RAFT optical-flow models](https://pytorch.org/vision/stable/models.html#optical-flow) — pretrained `raft_large`/`raft_small` for the learned-vs-classical flow comparison (Exercise 4).
- [MPI Sintel Flow benchmark](http://sintel.is.tue.mpg.de/) — the standard flow evaluation set, its metrics, and the color-coding convention for visualizing flow fields.
- Teed & Deng (2020), *RAFT* — paper + official code (github.com/princeton-vl/RAFT); read the correlation-volume and GRU-update sections alongside Lecture 4.
- Bertasius et al. (2021), *TimeSformer* — paper + code; the divided space-time attention ablation table is the clearest statement of the cost/accuracy trade in Lecture 5.
- [decord](https://github.com/dmlc/decord) / NVIDIA DALI docs — efficient video decoding to keep the GPU fed; the practical fix for the I/O bottleneck named in Lecture 3.
- *Dive into Deep Learning* (d2l.ai) and Stanford CS231n video-understanding notes — free, careful treatments of 3D CNNs, two-stream, and evaluation protocols to pair with Lecture 2.

> All external links are references, not endorsements. Prefer primary sources (papers, official docs) when they exist.
