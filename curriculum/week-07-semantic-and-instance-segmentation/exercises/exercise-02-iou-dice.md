# Exercise 2 — Implement IoU and Dice for masks

**Goal:** the honest segmentation metrics, by hand.

## Tasks

1. Implement `mask_iou(pred, gt)` and `dice(pred, gt)` for binary masks (boolean arrays). Verify the relationship `Dice = 2·IoU/(1+IoU)` on a test case.
2. Hand-draw (or use a tool to create) ground-truth masks for one class in 3–5 of your images.
3. Compute IoU and Dice between your model's predicted mask for that class and your ground truth.
4. Also compute pixel accuracy and show, on an image where the object is small, that pixel accuracy looks high while IoU is modest — the accuracy trap.

## Deliverable

A notebook with tested `mask_iou`/`dice`, the metric values on your images, and the demonstration that pixel accuracy overstates quality for small objects. This is why mIoU/Dice exist.
