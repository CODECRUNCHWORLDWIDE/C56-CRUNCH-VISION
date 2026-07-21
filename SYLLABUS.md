# C56 · Crunch Vision — Syllabus

> A free, open-source 12-week course that builds computer vision from the pixel up — filtering, features, CNNs, detection, segmentation, tracking, video, and Vision Transformers — every idea shown on real images before you trust a metric.

**School:** Artificial Intelligence · **Level:** Intermediate · **Duration:** ~288 hours over 12 weeks

## Prerequisites

- Comfortable Python: functions, classes, list/dict comprehensions, NumPy arrays. ([C1 Crunch Convos](../C1-Code-Crunch-Convos/) or equivalent.)
- Deep-learning basics — you can train a small network and know what a gradient and a loss are. [C53 Crunch Nets](../C53-CRUNCH-NETS/) is the ideal prerequisite; a solid grasp of backprop and PyTorch `nn.Module` covers it.
- High-school algebra and comfort with the idea of a matrix. We teach the vision-specific math (convolution, IoU) as we go.
- A machine with 8 GB RAM and Python 3.11+. A GPU is optional; every exercise runs on CPU.

## Grading & completion

You complete C56 · Crunch Vision by finishing the twelve weekly mini-projects and
the Week 12 capstone. There is no proctored exam; the quizzes are for you. Each
week's mini-project has a **Definition of Done** — meet it before moving on.

## Weekly plan

| # | Week | Core objectives |
|---|------|-----------------|
| 1 | Week 1 — Images, pixels, color & filtering | **Load and inspect** images as arrays: understand shape, dtype, channel order, and value range., **Explain** color spaces — RGB, grayscale, HSV — and convert between them deliberately., **Implement** 2-D convolution by hand and use it to blur, sharpen, and detect gradients. |
| 2 | Week 2 — Classical CV: edges, features, descriptors | **Build** a Canny edge detector and explain each of its stages., **Detect** corners with the Harris response and explain why corners are good features., **Extract** keypoints and descriptors (ORB or SIFT) and explain what a descriptor encodes. |
| 3 | Week 3 — Convolutional neural networks for vision | **Explain** why a fully-connected network is wrong for images and how convolution fixes it (weight sharing, locality)., **Implement** a convolution layer and pooling, and compute output shapes for any config., **Reason** about channels, receptive fields, and how depth builds a feature hierarchy. |
| 4 | Week 4 — Image classification end to end | **Build** a custom `Dataset`/`DataLoader` from an image-folder dataset with train/val/test splits., **Choose** and compute the right metrics — accuracy, per-class precision/recall, top-k, confusion matrix., **Regularize** deliberately with augmentation, weight decay, dropout, and early stopping. |
| 5 | Week 5 — Transfer learning for vision | **Explain** why features learned on ImageNet transfer to new tasks, and where the transfer breaks down., **Load** a pretrained backbone and adapt its classifier head to a new number of classes., **Apply** feature extraction (freeze the backbone) and full fine-tuning (train it all) and compare them. |
| 6 | Week 6 — Object detection | **Represent** detections as boxes + labels + scores, and compute Intersection-over-Union (IoU) by hand., **Implement** non-max suppression to remove duplicate boxes., **Explain** the detector design space: one-stage vs. two-stage, anchor-based vs. anchor-free. |
| 7 | Week 7 — Semantic & instance segmentation | **Distinguish** semantic, instance, and panoptic segmentation by what each labels., **Explain** the encoder–decoder (U-Net) architecture and why skip connections matter for masks., **Describe** how Mask R-CNN adds a mask branch to a detector for instance segmentation. |
| 8 | Week 8 — Keypoints, pose & object tracking | **Explain** keypoint estimation as per-keypoint heatmap regression, and pose as a set of connected keypoints., **Run** a pretrained pose estimator and interpret its keypoints and skeleton., **Distinguish** top-down vs. bottom-up pose approaches and their trade-offs. |
| 9 | Week 9 — Video understanding & optical flow | **Define** optical flow and estimate it, understanding the brightness-constancy assumption and its limits., **Explain** why temporal context is essential and how it changes what tasks become possible., **Compare** video architectures — frame aggregation, 3D CNNs, two-stream, and video Transformers. |
| 10 | Week 10 — Vision Transformers | **Explain** how an image becomes a sequence of patch tokens with positional embeddings., **Describe** self-attention and why it gives a global receptive field from the first layer., **Implement** patch embedding and reason about attention's quadratic cost. |
| 11 | Week 11 — Deploying vision on the edge | **Explain** the constraints of edge hardware — compute, memory, power, latency — and how they shape model choice., **Choose** an efficient architecture and understand the design tricks (depthwise-separable convs) that make it small., **Apply** quantization (and pruning/distillation) to shrink and speed up a model. |
| 12 | Week 12 — Capstone: a computer-vision system | **Frame** a real vision problem as a classification, detection, or segmentation task with a clear metric and a documented baseline., **Build and train** an appropriate model, applying transfer learning and the training toolbox deliberately., **Evaluate** honestly on held-out images with an error analysis, dataset-bias check, and named failure modes. |

## Capstone

In Week 12 you design, build, train, evaluate, and deploy one computer-vision system end to end on a problem and dataset you choose: frame the task (classification, detection, or segmentation), build a model that beats a documented baseline, evaluate it honestly on held-out images with an error analysis and named failure modes, then export it and serve it behind a small inference API with a model card. The deliverable is a repo someone else can clone, run on their own images, and trust.

---

_Part of [Code Crunch Worldwide](https://github.com/CODECRUNCHWORLDWIDE)._
