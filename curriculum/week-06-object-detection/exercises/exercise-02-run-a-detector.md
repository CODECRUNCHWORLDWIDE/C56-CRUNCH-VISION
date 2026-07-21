# Exercise 2 — Run a pretrained detector

**Goal:** get real detections on your own images.

## Tasks

1. Load a pretrained detector from torchvision (e.g. `fasterrcnn_resnet50_fpn`) or a YOLO model, with COCO weights.
2. Run it on several of your own photos. Filter predictions by a confidence threshold.
3. Draw the surviving boxes with class labels and scores on each image.
4. Experiment with the confidence threshold — show how a low threshold floods the image with false boxes and a high one misses objects. Confirm NMS is (already) applied.

## Deliverable

A notebook showing your images with drawn detections at a sensible confidence threshold, plus the low/high-threshold comparison. Note any obviously wrong or missed detections — that's the start of an error analysis.
