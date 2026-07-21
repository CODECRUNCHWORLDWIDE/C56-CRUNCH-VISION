# Mini-Project — A Segmentation System, Measured with IoU/Dice

## Brief

Build a segmentation system that labels pixels — semantic or instance — and evaluate it with the honest metrics, proving you understand dense prediction, encoder–decoders, and why masks are measured with IoU/Dice.

## Requirements

1. **Task & model:** choose semantic (U-Net/DeepLab/FCN) or instance (Mask R-CNN). Run a pretrained model, or fine-tune/train a U-Net on a small dataset.
2. **Inference:** overlay colored masks on your images — per-class for semantic, per-instance for instance.
3. **Metrics:** your own `mask_iou` and `dice`; report mean IoU and/or Dice against ground-truth masks, per class. Explicitly show why you do *not* rely on pixel accuracy.
4. **Error analysis:** a visual gallery of failures — blurry boundaries, missed thin/small objects, merged instances — with masks overlaid.
5. **README:** the task chosen and why, the metrics defined, reproduce steps, and honest limitations (including ground-truth boundary ambiguity).

## Stretch

- Train a U-Net with a skip-connection ablation (Challenge 1).
- Compare Dice vs. cross-entropy loss on an imbalanced task (Challenge 2).

## What you're proving

You can produce and honestly measure pixel-level predictions — the most precise form of image understanding. You've now covered classification, detection, and segmentation: the three pillars of static-image vision. Next week images start to *move* — keypoints, pose, and tracking objects across frames.
