# Exercise 2 — Implement IoU, Dice, and mIoU, and expose the accuracy trap

**Goal:** the honest segmentation metrics, by hand, with the multi-class aggregation done correctly.

## Tasks

1. Implement `mask_iou(pred, gt)` and `dice(pred, gt)` for binary boolean masks. Handle the empty-mask edge case
   explicitly (define IoU=1 when both are empty, IoU=0 when exactly one is). **Verify** `Dice = 2·IoU/(1+IoU)` numerically
   on several random masks.
2. Implement `mean_iou(pred_labels, gt_labels, num_classes)` that accumulates per-class TP/FP/FN **across a set of
   images** and only then divides — the correct dataset-level aggregation — and returns both per-class IoU and the mean.
   Contrast it in a comment with the wrong approach (averaging per-image IoU) and say when they diverge.
3. Validate your `mean_iou` against a reference (e.g. `torchmetrics.JaccardIndex` or a hand-computed 3-pixel toy case)
   with `assert`/`allclose`.
4. Hand-draw (or use a tool to create) ground-truth masks for one class in 3–5 of your images.
5. Compute IoU and Dice between the model's predicted mask for that class and your ground truth. Then compute **pixel
   accuracy** and demonstrate, on an image where the object is small, that pixel accuracy reads high (>0.95) while IoU is
   modest (say 0.5) — the accuracy trap, quantified.

## Deliverable

A notebook with tested `mask_iou`/`dice`/`mean_iou` (including the empty-mask and aggregation edge cases), a passing
validation against a reference, the metric values on your images, and the numeric demonstration that pixel accuracy
overstates quality for small objects. This is *why* mIoU/Dice exist.
