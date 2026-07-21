# Exercise 3 — Measure mAP on hand-labeled data

**Goal:** compute and interpret the detection metric.

## Tasks

1. Hand-label ground-truth boxes for ~15–20 of your images (a few objects each) — or use a small labeled subset of a public set.
2. Run your detector and, using your IoU function, match predictions to ground truth at IoU ≥ 0.5. Count true positives, false positives, and false negatives.
3. Compute per-class precision, recall, and Average Precision (area under the PR curve), then mAP@0.5. You may use `torchmetrics` to cross-check.
4. Recompute at IoU ≥ 0.75 and compare — what does the drop tell you about localization tightness?

## Deliverable

A notebook reporting per-class AP and mAP at two IoU thresholds, with a short interpretation of the 0.5-vs-0.75 gap. If mAP is near zero but boxes look right, your matching or class indexing is off — debug it.
