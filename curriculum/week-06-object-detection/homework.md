# Week 6 — Homework

Cement detection's vocabulary and metrics before segmentation.

## Tasks

- Compute IoU by hand for two boxes you define, then verify with your code.
- Write, in your own words, the greedy NMS algorithm and why it runs per class.
- Explain the difference between mAP@0.5 and mAP@[.5:.95] and when each is appropriate.
- Read the torchvision detection tutorial and the YOLO docs (in resources); summarize one-stage vs. two-stage trade-offs.

## Definition of done

A committed project that runs (or fine-tunes) an object detector on your own images, applies your hand-written IoU + NMS, visualizes detections with labels and scores, and evaluates against hand-labeled ground truth with per-class AP and mAP at IoU 0.5 (and ideally 0.75), including an error analysis of failure modes.

Submit by committing your work to your course repo under `week-06/`.
