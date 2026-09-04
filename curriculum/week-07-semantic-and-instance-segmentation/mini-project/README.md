# Mini-Project — A Measured Segmentation System: Semantic, Instance, and the Honest Metrics

## Brief

Build a segmentation system that labels pixels — semantic and/or instance — run and/or train a model, and **evaluate it
with the honest metrics implemented from scratch**, proving you understand dense prediction, the architectures, the loss
choice, and why masks are measured with IoU/Dice/PQ rather than pixel accuracy.

## Requirements

1. **Task & model.** Choose semantic (U-Net / DeepLabV3 / SegFormer) and/or instance (Mask R-CNN) — at least one run
   *and* one trained/fine-tuned model if you can. State which task you chose and why it fits your question (do you need to
   separate individual objects, or is a class map enough?).
2. **Inference & overlays.** Overlay colored masks on your own images: per-class for semantic, per-instance for
   instance. Include several hard cases, not just easy ones.
3. **Metrics from scratch.** Implement and unit-test your own `mask_iou`, `dice`, and a dataset-level `mean_iou` that
   accumulates per-class TP/FP/FN before dividing. Validate against a reference (`torchmetrics`). Report **per-class IoU
   and mIoU and/or Dice** against ground-truth masks you create. **Explicitly show** why you do not rely on pixel
   accuracy, with the small-object demonstration from Exercise 2.
4. **Loss justification.** State which loss you used (or would use) and *why*, in terms of your data's class balance and
   the metric you report (Lecture 4). If you trained, show at least a two-way loss comparison (e.g. CE vs. CE+Dice) with
   foreground metrics.
5. **Error analysis.** A visual gallery of failures — blurry/leaky boundaries, missed thin or small objects, merged or
   split instances — each with the mask overlaid and a named likely cause.
6. **Honest limitations & model card.** A README covering: task chosen and why; metrics defined; reproduce steps; the
   loss rationale; and honest limitations — including the **ground-truth boundary ambiguity / annotator-disagreement
   ceiling** and, if relevant to your domain, calibration and distribution-shift concerns.

## Stretch

- Train a U-Net with the skip-connection ablation and Boundary IoU (Challenge 1).
- Compare CE / Dice / Lovász-Softmax on an imbalanced task (Challenge 2).
- Run a panoptic model and compute PQ = SQ × RQ by hand (Challenge 3).
- Add a reliability diagram and comment on calibration.

## What you're proving

You can produce *and honestly measure* pixel-level predictions — the most precise form of static image understanding —
choosing the architecture and loss deliberately and reporting metrics that cannot be gamed by the majority class. With
classification, detection, and now segmentation, you have the three pillars of static-image vision. Next week the images
start to *move*: keypoints, pose, optical flow, and tracking objects across frames.
