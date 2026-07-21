# Week 6 — Quiz

Ten questions. Answer key below.

**1. Object detection outputs, per object:**

- A. A bounding box, a class label, and a confidence score
- B. Only a pixel mask
- C. A single number for the whole image
- D. Only a class label

**2. Intersection over Union (IoU) is defined as:**

- A. Overlap area minus union area
- B. Overlap area divided by union area
- C. Union divided by overlap
- D. The larger box's area

**3. Non-maximum suppression removes:**

- A. Low-resolution images
- B. Background pixels
- C. The final layer
- D. Duplicate overlapping boxes for the same object, keeping the highest-scoring

**4. A two-stage detector (e.g. Faster R-CNN):**

- A. Predicts boxes in one pass
- B. Proposes regions, then classifies and refines them — accurate but slower
- C. Uses no convolutions
- D. Only classifies whole images

**5. One-stage detectors (e.g. YOLO) are favored for:**

- A. Not needing any training
- B. Grayscale images only
- C. Maximum accuracy on tiny objects
- D. Real-time / edge use because they are fast (single pass)

**6. Anchor-free detectors differ by:**

- A. Predicting boxes directly (e.g. from centers) instead of refining preset anchor boxes
- B. Only detecting one class
- C. Requiring anchors of every size
- D. Using no neural network

**7. The background/foreground imbalance in detection is often handled by:**

- A. Focal loss (down-weighting easy background examples)
- B. Ignoring it
- C. Using MSE loss
- D. Removing all background

**8. Average Precision (AP) for a class is:**

- A. The area under that class's precision–recall curve
- B. The number of boxes
- C. The IoU threshold
- D. The single best prediction's score

**9. mAP@[.5:.95] (COCO) differs from mAP@0.5 by:**

- A. Ignoring classification
- B. Using no IoU
- C. Averaging AP over IoU thresholds 0.5–0.95, rewarding tighter localization
- D. Being always higher

**10. A detector with high mAP@0.5 but low mAP@0.75 is:**

- A. Perfectly localized
- B. Not detecting anything
- C. Overfit to background
- D. Finding objects but localizing loosely

---

## Answer key

1. **A. A bounding box, a class label, and a confidence score** — Detection localizes (box) and classifies (label + score) each object.
2. **B. Overlap area divided by union area** — IoU = intersection ÷ union, ranging 0 (no overlap) to 1 (identical).
3. **D. Duplicate overlapping boxes for the same object, keeping the highest-scoring** — Greedy NMS keeps top-score boxes and deletes their high-IoU duplicates, per class.
4. **B. Proposes regions, then classifies and refines them — accurate but slower** — Region proposals then per-region classification: typically more accurate, slower.
5. **D. Real-time / edge use because they are fast (single pass)** — A single dense forward pass makes them fast enough for live video and edge devices.
6. **A. Predicting boxes directly (e.g. from centers) instead of refining preset anchor boxes** — They drop predefined anchors and their tuning, predicting boxes directly.
7. **A. Focal loss (down-weighting easy background examples)** — Focal loss reduced one-stage detectors' overwhelming easy-negative dominance.
8. **A. The area under that class's precision–recall curve** — AP summarizes the precision/recall trade-off as the PR-curve area.
9. **C. Averaging AP over IoU thresholds 0.5–0.95, rewarding tighter localization** — COCO mAP averages across stricter IoU thresholds, so it demands precise boxes.
10. **D. Finding objects but localizing loosely** — It matches at a loose IoU but boxes aren't tight enough for the stricter threshold.
