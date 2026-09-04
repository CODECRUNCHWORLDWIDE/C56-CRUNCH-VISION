# Week 6 — Quiz

Fifteen questions spanning box geometry and the IoU/GIoU family, greedy vs. Soft-NMS, the detector design map, label assignment and focal loss, the exact mAP protocol, and DETR set prediction. Attempt each before the answer key — several test the distinctions (metric vs. loss, mAP@0.5 vs. [.5:.95], one-to-many vs. one-to-one) that separate a working understanding from a vocabulary.

**1. Object detection outputs, per object instance:**

- A. only a pixel-level segmentation mask
- B. a single class label for the whole image
- C. the image's dominant color
- D. a bounding box, a class label, and a confidence score

<details>
<summary>Answer</summary>

**D. a bounding box, a class label, and a confidence score** — Detection localizes (box) and classifies (label + score) each instance; a variable-length set of these is the output.

</details>

**2. Intersection over Union (IoU) is defined as:**

- A. overlap area minus union area
- B. overlap area divided by union area
- C. union area divided by overlap area
- D. the larger box's area divided by the smaller's

<details>
<summary>Answer</summary>

**B. overlap area divided by union area** — IoU = intersection ÷ union, ranging 0 (disjoint) to 1 (identical), scale-invariant and symmetric.

</details>

**3. The `max(0, ...)` clamp in an IoU implementation exists to:**

- A. speed up the computation
- B. normalize coordinates to [0,1]
- C. convert corner form to center form
- D. prevent two negative side lengths from multiplying into a spurious positive intersection for disjoint boxes

<details>
<summary>Answer</summary>

**D. prevent two negative side lengths from multiplying into a spurious positive intersection for disjoint boxes** — Without clamping, disjoint boxes give negative iw and ih whose product is positive — a classic false-overlap bug.

</details>

**4. Plain IoU is a poor *regression loss* (as opposed to a good metric) because:**

- A. it is too slow to compute
- B. for two disjoint boxes it is 0 for every configuration, so its gradient is zero and cannot pull boxes together
- C. it only works on square boxes
- D. it requires the softmax function

<details>
<summary>Answer</summary>

**B. for two disjoint boxes it is 0 for every configuration, so its gradient is zero and cannot pull boxes together** — IoU's zero-gradient dead zone for non-overlapping boxes is exactly what GIoU/DIoU/CIoU were designed to fix.

</details>

**5. Greedy non-maximum suppression:**

- A. keeps the highest-scoring box and deletes remaining boxes whose IoU with it exceeds a threshold, per class
- B. averages all overlapping boxes into one
- C. removes low-resolution images from the batch
- D. deletes background pixels before classification

<details>
<summary>Answer</summary>

**A. keeps the highest-scoring box and deletes remaining boxes whose IoU with it exceeds a threshold, per class** — Greedy NMS sorts by score, keeps the top box, and suppresses its high-IoU duplicates, repeating per class.

</details>

**6. Soft-NMS differs from greedy NMS by:**

- A. decaying an overlapping box's score (e.g. by exp(-IoU²/σ)) instead of deleting it outright, improving recall in crowds
- B. using no IoU at all
- C. requiring a two-stage detector
- D. running before the network instead of after

<details>
<summary>Answer</summary>

**A. decaying an overlapping box's score (e.g. by exp(-IoU²/σ)) instead of deleting it outright, improving recall in crowds** — Soft-NMS replaces hard deletion with score decay, so a heavily overlapping second real object can still survive at lower rank.

</details>

**7. A two-stage detector (e.g. Faster R-CNN):**

- A. uses a Region Proposal Network to suggest regions, then classifies and refines each — accurate but slower
- B. predicts all boxes in a single dense pass
- C. only classifies whole images without localization
- D. uses no convolutions

<details>
<summary>Answer</summary>

**A. uses a Region Proposal Network to suggest regions, then classifies and refines each — accurate but slower** — An RPN proposes ~1000 regions, then a second head classifies and box-refines them: typically more accurate, slower.

</details>

**8. One-stage detectors (e.g. YOLO, RetinaNet) are favored for real-time/edge use because:**

- A. they only work on grayscale images
- B. a single dense forward pass makes them fast enough for live video
- C. they are always the most accurate on tiny objects
- D. they need no training data

<details>
<summary>Answer</summary>

**B. a single dense forward pass makes them fast enough for live video** — One pass over a dense grid gives real-time throughput, which is why YOLO dominates video and edge deployment.

</details>

**9. Anchor-free detectors (e.g. FCOS) differ from anchor-based ones by:**

- A. using no neural network
- B. predicting boxes directly (e.g. distances from a center location) instead of refining preset reference boxes
- C. detecting only a single class
- D. requiring one anchor of every possible size

<details>
<summary>Answer</summary>

**B. predicting boxes directly (e.g. distances from a center location) instead of refining preset reference boxes** — Anchor-free methods drop predefined anchors and their tuning, predicting boxes directly from locations/centers.

</details>

**10. In detection training, 'label assignment' refers to:**

- A. choosing the dataset's class names
- B. assigning a learning rate to each layer
- C. labeling the test set
- D. deciding which predictions/anchors are positives (responsible for which ground-truth object) and which are background

<details>
<summary>Answer</summary>

**D. deciding which predictions/anchors are positives (responsible for which ground-truth object) and which are background** — Assignment maps each candidate prediction to a GT (positive) or background (negative), defining what the loss means.

</details>

**11. The extreme foreground/background imbalance in dense one-stage detectors is best described as:**

- A. roughly balanced, ~1:1
- B. background outnumbering foreground by roughly 1000:1, so easy negatives dominate an ordinary cross-entropy gradient
- C. foreground outnumbering background
- D. irrelevant to training

<details>
<summary>Answer</summary>

**B. background outnumbering foreground by roughly 1000:1, so easy negatives dominate an ordinary cross-entropy gradient** — ~100k anchors vs. tens of foreground gives ~1000:1; summed easy-negative loss swamps foreground under plain CE.

</details>

**12. Focal loss FL(p_t) = -(1-p_t)^γ log(p_t) reduces the imbalance problem by:**

- A. removing all background examples from the batch
- B. switching classification to mean squared error
- C. increasing the learning rate on background
- D. down-weighting well-classified (easy) examples via the (1-p_t)^γ factor while leaving hard examples near full weight

<details>
<summary>Answer</summary>

**D. down-weighting well-classified (easy) examples via the (1-p_t)^γ factor while leaving hard examples near full weight** — As p_t→1 the modulating factor →0, silencing easy negatives; hard examples (small p_t) keep their gradient — no sampling needed.

</details>

**13. At γ = 2, an easy example with p_t = 0.9 has its loss scaled by approximately:**

- A. 1 (unchanged)
- B. 0.01 — a roughly 100× reduction
- C. 10 (amplified)
- D. 0 exactly

<details>
<summary>Answer</summary>

**B. 0.01 — a roughly 100× reduction** — (1 - 0.9)^2 = 0.1^2 = 0.01, a ~100× down-weighting of the easy example relative to plain cross-entropy.

</details>

**14. In computing AP, each ground-truth box may be matched to:**

- A. exactly the two highest-scoring predictions
- B. no predictions ever
- C. any number of predictions
- D. at most one prediction; a second prediction on the same object is a duplicate false positive

<details>
<summary>Answer</summary>

**D. at most one prediction; a second prediction on the same object is a duplicate false positive** — One-GT-per-prediction matching is how the metric penalizes missing NMS: duplicates become false positives.

</details>

**15. COCO's primary metric mAP@[.5:.95] differs from mAP@0.5 by:**

- A. using no IoU threshold
- B. averaging AP over ten IoU thresholds from 0.50 to 0.95, rewarding tighter localization
- C. always producing a higher number than mAP@0.5
- D. ignoring the class labels entirely

<details>
<summary>Answer</summary>

**B. averaging AP over ten IoU thresholds from 0.50 to 0.95, rewarding tighter localization** — Averaging over stricter IoU thresholds demands precise boxes, so mAP@[.5:.95] ≤ mAP@0.5 and the two are not comparable.

</details>

**16. DETR removes both anchors and NMS because its bipartite Hungarian matching is:**

- A. one-to-one, so each ground truth is matched to exactly one prediction and duplicates are trained toward 'no object'
- B. one-to-many, rewarding redundant predictions
- C. applied only at inference, not training
- D. based on hand-tuned anchor scales

<details>
<summary>Answer</summary>

**A. one-to-one, so each ground truth is matched to exactly one prediction and duplicates are trained toward 'no object'** — One-to-one assignment penalizes duplicates during training, so the network learns not to emit them — NMS becomes unnecessary.

</details>

---
