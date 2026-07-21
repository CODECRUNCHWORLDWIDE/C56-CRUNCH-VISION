# Week 6 — Object detection

> **Goal:** by Sunday you can explain how detectors turn an image into boxes-with-labels, implement IoU and non-max suppression by hand, run a pretrained detector on your own images, and evaluate detection quality with mean Average Precision.

Classification answers 'what is this image?'. Detection answers the harder, more useful question: 'what objects are here, and *where*?' — outputting a labeled bounding box for each. This week you learn the machinery that makes that possible: **bounding boxes and IoU**, **non-max suppression** (which you'll build by hand — it's the same idea as Canny's NMS from Week 2), the **anchor vs. anchor-free** and **one-stage vs. two-stage** design axes, and the metric that rules detection benchmarks, **mAP**. You'll run a real detector and evaluate it honestly.

## Learning objectives

By the end of this week, you will be able to:

- **Represent** detections as boxes + labels + scores, and compute Intersection-over-Union (IoU) by hand.
- **Implement** non-max suppression to remove duplicate boxes.
- **Explain** the detector design space: one-stage vs. two-stage, anchor-based vs. anchor-free.
- **Run** a pretrained detector (Faster R-CNN or a YOLO family model) on your own images.
- **Evaluate** detection with precision/recall at IoU thresholds and mean Average Precision (mAP).

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Bounding boxes, IoU, and the detection task](lecture-notes/01-boxes-and-iou.md)
2. [Lecture 2 — Non-max suppression & detector architectures](lecture-notes/02-nms-and-architectures.md)
3. [Lecture 3 — Training detectors & measuring mAP](lecture-notes/03-training-and-map.md)

**Exercises**

1. [Exercise 1 — Implement IoU and non-max suppression](exercises/exercise-01-iou-and-nms.md)
2. [Exercise 2 — Run a pretrained detector](exercises/exercise-02-run-a-detector.md)
3. [Exercise 3 — Measure mAP on hand-labeled data](exercises/exercise-03-measure-map.md)

**Challenges**

1. [Challenge 1 — Fine-tune a detector on a custom class](challenges/challenge-01-finetune-a-detector.md)
2. [Challenge 2 — The speed–accuracy frontier](challenges/challenge-02-speed-accuracy.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 7.

## Deliverable

A notebook that runs a pretrained object detector on a set of your own images, draws the predicted boxes with labels and scores, applies your own IoU + non-max-suppression implementation, and reports mAP against a small hand-labeled ground truth.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
