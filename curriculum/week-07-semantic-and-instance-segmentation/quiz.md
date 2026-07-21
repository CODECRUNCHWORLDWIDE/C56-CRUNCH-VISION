# Week 7 — Quiz

Ten questions. Answer key below.

**1. Semantic segmentation assigns:**

- A. A box per object
- B. An instance ID to every pixel
- C. A class to every pixel, merging same-class objects
- D. One label to the whole image

**2. Instance segmentation differs from semantic by:**

- A. Separating each object with its own mask and instance ID
- B. Only working on backgrounds
- C. Using boxes only
- D. Labeling no pixels

**3. Panoptic segmentation:**

- A. Only detects boxes
- B. Is the same as classification
- C. Labels every pixel, giving instance IDs to things and a class to stuff
- D. Ignores stuff

**4. An encoder–decoder is used for segmentation because it:**

- A. Only downsamples
- B. Needs no convolutions
- C. Builds deep 'what' features (encoder) then recovers spatial 'where' at full resolution (decoder)
- D. Avoids pretrained backbones

**5. U-Net's skip connections:**

- A. Speed up training only
- B. Add classes
- C. Remove the decoder
- D. Copy high-resolution encoder features to the decoder so masks have sharp boundaries

**6. Mask R-CNN produces instance masks by:**

- A. Segmenting the whole image at once
- B. Using only edges
- C. Adding a per-object mask branch on top of a detector (Faster R-CNN)
- D. Classifying pixels randomly

**7. Pixel accuracy is a poor segmentation metric because:**

- A. It is too slow
- B. It equals IoU
- C. It needs masks
- D. Background/majority pixels dominate, hiding missed small objects

**8. Mean IoU (mIoU) averages IoU:**

- A. Over classes, weighting each class equally
- B. Over epochs
- C. Over images only
- D. Over pixels

**9. The Dice coefficient is favored in medical imaging and:**

- A. Is often used as a differentiable loss that handles class imbalance well
- B. Equals pixel accuracy
- C. Cannot be a loss
- D. Ignores overlap

**10. Instance segmentation is evaluated with:**

- A. Box IoU only
- B. Dice on the whole image
- C. Mask AP / mask mAP (matching by mask IoU)
- D. Pixel accuracy only

---

## Answer key

1. **C. A class to every pixel, merging same-class objects** — Semantic = per-pixel class; all cars share the 'car' label, not separated.
2. **A. Separating each object with its own mask and instance ID** — Instance separates object #1 from object #2 (countable 'things'), unlike semantic.
3. **C. Labels every pixel, giving instance IDs to things and a class to stuff** — It unifies semantic + instance for a complete per-pixel scene description.
4. **C. Builds deep 'what' features (encoder) then recovers spatial 'where' at full resolution (decoder)** — Segmentation needs both semantic depth and spatial precision.
5. **D. Copy high-resolution encoder features to the decoder so masks have sharp boundaries** — Skips fuse deep 'what' with shallow 'where', giving boundary-hugging masks.
6. **C. Adding a per-object mask branch on top of a detector (Faster R-CNN)** — It detects each object, then predicts a mask inside each box — 'detect then mask'.
7. **D. Background/majority pixels dominate, hiding missed small objects** — Labeling everything background can score high while missing every object.
8. **A. Over classes, weighting each class equally** — Per-class averaging stops rare classes from hiding behind common ones.
9. **A. Is often used as a differentiable loss that handles class imbalance well** — Dice = 2·IoU/(1+IoU); it's a common imbalance-robust loss for tiny targets.
10. **C. Mask AP / mask mAP (matching by mask IoU)** — Mask mAP blends detection quality with per-object mask overlap.
