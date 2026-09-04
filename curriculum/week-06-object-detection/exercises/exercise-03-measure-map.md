# Exercise 3 — Compute COCO-style mAP by hand and validate it against torchmetrics

**Goal:** implement the detection metric end to end so you trust — and can debug — every number.

## Tasks

1. Hand-label ground-truth boxes for ~15-20 of your images (a few objects each), or use a small labeled subset
   of a public set. Store boxes + class ids in a simple structure.
2. Implement the full AP pipeline yourself: greedy IoU matching in score order, one-GT-per-prediction,
   TP/FP/FN counting, the accumulated precision-recall curve, and **all-point** (COCO-style) interpolation.
   Compute per-class AP and mAP@0.5.
3. **Cross-check.** Run `torchmetrics.detection.MeanAveragePrecision` on the identical predictions and ground
   truth and confirm your mAP@0.5 agrees to a few decimals. If it does not, the usual culprits are: class-index
   mismatch, a corner/center box-format mismatch, or double-matching a GT — debug until they agree.
4. Recompute at IoU ≥ 0.75, then average AP over the ten thresholds 0.50:0.05:0.95 to get **mAP@[.5:.95]**.
   Report all three (0.5, 0.75, [.5:.95]) plus AP_S/AP_M/AP_L if you have enough small objects.
5. Write a short interpretation of the 0.5-vs-0.75 gap for *your* detector.

## Deliverable

A notebook reporting your hand-computed per-class AP and mAP at 0.5, 0.75, and [.5:.95], agreeing with
torchmetrics at 0.5, with a paragraph interpreting the localization-tightness gap. If mAP is near zero but the
drawn boxes look right, your matching or class indexing is wrong — that debugging *is* the exercise.
