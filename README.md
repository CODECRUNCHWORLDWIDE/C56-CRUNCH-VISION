# C56 · Crunch Vision

> A free, open-source 12-week course that builds computer vision from the pixel up — filtering, features, CNNs, detection, segmentation, tracking, video, and Vision Transformers — every idea shown on real images before you trust a metric.

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![Python · PyTorch · OpenCV](https://img.shields.io/badge/stack-Python_·_PyTorch_·_OpenCV-14B8A6.svg)](#stack)
[![Built in the open](https://img.shields.io/badge/built-in%20the%20open-14B8A6.svg)](https://github.com/CODECRUNCHWORLDWIDE)

C56 is a Tier-2 computer-vision course, sized full (12 weeks) because it walks the whole road — from "what actually is a pixel, and what is a color?" to building, training, and deploying classifiers, detectors, segmenters, trackers, and Vision Transformers on real images and video. You will implement **convolution, edge detection, and non-max suppression by hand** before you lean on the library, so no layer is ever a black box. Everything is in **Python + PyTorch + OpenCV**, runs on a laptop CPU (a GPU only makes it faster), and pairs naturally with [C53 Crunch Nets](../C53-CRUNCH-NETS/) for the deep-learning foundations, [C5 Crunch AI & Data Science](../C5-CRUNCH-AI-DATA-SCIENCE/) for the data work, and [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/) for shipping what you build.

---

## Pathway summary

- **Full-time:** 12 weeks · ~24 hrs/week · ~288 hours
- **Working-engineer pace:** 6 months · ~12 hrs/week
- **Evening pace:** 12 months · ~6 hrs/week

See [`SYLLABUS.md`](SYLLABUS.md).

---

## What you will be able to do at the end of 12 weeks

- **Reason about images as data:** explain pixels, color spaces, channels, and resolution, and load, transform, and visualize images fluently.
- **Filter and transform images:** implement convolution, blurring, sharpening, and gradients by hand, and know when to reach for the classical tool.
- **Extract classical features:** find edges, corners, and keypoints, and match images with descriptors — the pre-deep-learning toolkit that still ships in production.
- **Build CNNs for vision:** implement convolution and pooling, then design and train convolutional classifiers that generalize.
- **Use transfer learning:** fine-tune pretrained backbones on your own small datasets and know when to freeze vs. fine-tune.
- **Detect objects:** understand anchors, IoU, and non-max suppression, and train and evaluate a detector with mAP.
- **Segment images:** build semantic and instance segmentation and measure them with the right metrics.
- **Track and understand motion:** estimate keypoints and pose, track objects across frames, and reason about optical flow and video models.
- **Build Vision Transformers:** implement patch embedding and self-attention for images and compare ViTs to CNNs honestly.
- **Deploy vision on the edge:** export, quantize, and benchmark a model to run in real time on constrained hardware.
- **Do it honestly:** measure on held-out images, report failure modes and dataset bias, and respect image licenses, privacy, and consent.

---

## The 12 weeks

| # | Week | Focus |
|---|------|-------|
| 1 | [Week 1 — Images, pixels, color & filtering](curriculum/week-01-images-pixels-color-and-filtering/) | Load images as tensors, understand color spaces, and filter by hand with convolution. |
| 2 | [Week 2 — Classical CV: edges, features, descriptors](curriculum/week-02-classical-cv-edges-features-descriptors/) | Detect edges and corners, describe keypoints, and match two images — the pre-deep toolkit. |
| 3 | [Week 3 — Convolutional neural networks for vision](curriculum/week-03-convolutional-neural-networks-for-vision/) | Build convolution and pooling layers, then assemble and train a small CNN. |
| 4 | [Week 4 — Image classification end to end](curriculum/week-04-image-classification-end-to-end/) | Build a real classification pipeline: datasets, metrics, regularization, and diagnosis. |
| 5 | [Week 5 — Transfer learning for vision](curriculum/week-05-transfer-learning-for-vision/) | Stand on pretrained backbones: feature extraction, fine-tuning, and when to do which. |
| 6 | [Week 6 — Object detection](curriculum/week-06-object-detection/) | Find and localize objects: boxes, IoU, non-max suppression, anchors, and mAP. |
| 7 | [Week 7 — Semantic & instance segmentation](curriculum/week-07-semantic-and-instance-segmentation/) | Label every pixel: semantic vs. instance, encoder–decoders, and the right metrics. |
| 8 | [Week 8 — Keypoints, pose & object tracking](curriculum/week-08-keypoints-pose-and-object-tracking/) | Locate keypoints, estimate human pose, and follow objects across video frames. |
| 9 | [Week 9 — Video understanding & optical flow](curriculum/week-09-video-understanding-and-optical-flow/) | Model motion and time: optical flow, action recognition, and temporal architectures. |
| 10 | [Week 10 — Vision Transformers](curriculum/week-10-vision-transformers/) | Patches, self-attention for images, and how ViTs compare to CNNs honestly. |
| 11 | [Week 11 — Deploying vision on the edge](curriculum/week-11-deploying-vision-on-the-edge/) | Make a model small and fast: efficient architectures, quantization, export, and benchmarking. |
| 12 | [Week 12 — Capstone: a computer-vision system](curriculum/week-12-capstone-a-computer-vision-system/) | One vision system, end to end, on a problem you choose — and defend it. |

---

## Stack

**Python 3.11+**, **PyTorch 2.x** + **torchvision**, **OpenCV**, **NumPy**, **Pillow**, **matplotlib**, and a notebook environment (Jupyter or VS Code). Datasets are small and public (MNIST, CIFAR-10, Oxford Pets, a handful of COCO images, short clips) so everything runs on a CPU in minutes; a GPU (local or a free cloud notebook) is optional and only speeds training up. No paid APIs, no proprietary tools.

## How each week is structured

Every week ships the same parts: **lecture notes** (the ideas), **exercises**
(guided, one concept each), **challenges** (open-ended, you make the calls), a
**mini-project** (the week's real build), a **quiz** (check yourself), and
**homework + resources**. Do them in that order.

## Ethics & honest practice

Vision is the most privacy-laden branch of AI — it looks at people, places, and documents. Four rules run through this course: **(1) Measure on images the model has never seen** — a number on training data is not a result. **(2) Report failure modes and dataset bias** — every model card names where the model breaks and for whom, and vision datasets are notoriously skewed by geography, skin tone, and lighting. **(3) Respect provenance, licenses, and consent** — you only train on images you are allowed to use, and faces and personal data deserve special care. **(4) Do not build surveillance you would not want aimed at you** — capability without conscience is not skill.

---

_Part of [Code Crunch Worldwide](https://github.com/CODECRUNCHWORLDWIDE) — free, open-source, built in the open._
