# Week 7 — Quiz

Fifteen questions spanning the three tasks, the receptive-field-vs-resolution tension and its architectural escapes, RoIAlign, the IoU/Dice/PQ derivations, the loss zoo, query-based segmenters, and high-stakes evaluation. Attempt each before the answer key.

**1. The output *space* of instance segmentation differs from semantic segmentation because instance segmentation predicts:**

- A. only bounding boxes, no pixel masks
- B. a class for every pixel including all stuff regions, with no instance IDs
- C. a single integer label map with one class per pixel
- D. a variable-length set of (class, binary-mask) pairs, one per object

<details>
<summary>Answer</summary>

**D. a variable-length set of (class, binary-mask) pairs, one per object** — Instance segmentation emits a set of per-object masks (like detection, variable length); semantic emits one class map that merges same-class objects.

</details>

**2. Panoptic segmentation's defining structural constraint on its output is that it is:**

- A. allowed to leave stuff pixels unlabeled
- B. identical to running semantic and instance models independently
- C. a gap-free, non-overlapping partition: every pixel gets exactly one (class, id) label
- D. a set of possibly-overlapping instance masks with background ignored

<details>
<summary>Answer</summary>

**C. a gap-free, non-overlapping partition: every pixel gets exactly one (class, id) label** — Panoptic requires every pixel to have exactly one label (things instanced, stuff classed) with no overlaps or gaps — which is why it needed its own coherent metric (PQ).

</details>

**3. A classification backbone at 1/32 resolution cannot directly produce pixel-precise masks because:**

- A. convolutions cannot be applied to segmentation
- B. softmax is undefined over spatial dimensions
- C. its receptive field is too small to see whole objects
- D. downsampling has discarded the spatial detail a boundary needs; a 7×7 grid cannot localize an edge

<details>
<summary>Answer</summary>

**D. downsampling has discarded the spatial detail a boundary needs; a 7×7 grid cannot localize an edge** — The receptive field is large but resolution is destroyed — you cannot recover a pixel-precise boundary from a coarse grid. This is the receptive-field-vs-resolution tension.

</details>

**4. U-Net's skip connections are decisive because they:**

- A. concatenate shallow high-resolution 'where' features onto deep 'what' features in the decoder, so masks hug boundaries
- B. replace the decoder entirely with the encoder
- C. reduce the number of parameters the decoder needs
- D. only speed up training without affecting mask quality

<details>
<summary>Answer</summary>

**A. concatenate shallow high-resolution 'where' features onto deep 'what' features in the decoder, so masks hug boundaries** — Deep features know 'what' but lost 'where'; skips carry sharp shallow features across so the decoder can fuse both — the reason masks are sharp rather than blobby.

</details>

**5. DeepLab's atrous (dilated) convolution enlarges the receptive field without losing resolution by:**

- A. inserting zeros between kernel taps so a 3×3 kernel covers a larger field at stride 1, with no extra weights
- B. cropping the feature map to the object
- C. using a fully-connected layer at the output
- D. adding more downsampling stages

<details>
<summary>Answer</summary>

**A. inserting zeros between kernel taps so a 3×3 kernel covers a larger field at stride 1, with no extra weights** — Dilation rate r spaces the kernel taps to cover a (2r+1)×(2r+1) field using only 9 weights and stride 1, so resolution is preserved — the opposite escape from encoder-decoder upsampling.

</details>

**6. Mask R-CNN's RoIAlign improved mask quality over RoIPool chiefly by:**

- A. removing the coordinate quantizations and sampling features at exact fractional RoI locations via bilinear interpolation
- B. adding non-maximum suppression to the mask branch
- C. using a larger backbone
- D. predicting masks at the full image resolution directly

<details>
<summary>Answer</summary>

**A. removing the coordinate quantizations and sampling features at exact fractional RoI locations via bilinear interpolation** — RoIPool quantizes RoI coordinates twice, misaligning features by up to ~1–2 pixels — tolerable for a box, ruinous for a pixel mask. RoIAlign's bilinear sub-pixel sampling fixes it.

</details>

**7. Given IoU = 0.5, the Dice coefficient equals:**

- A. 2/3 (from Dice = 2·IoU/(1+IoU))
- B. 1.0
- C. 0.5 (they are always equal)
- D. 0.25

<details>
<summary>Answer</summary>

**A. 2/3 (from Dice = 2·IoU/(1+IoU))** — Dice = 2(0.5)/(1+0.5) = 1/1.5 = 2/3. Dice ≥ IoU on [0,1], and the map is strictly increasing so they rank models identically.

</details>

**8. mIoU averages IoU over classes with equal weight *on purpose* because:**

- A. it is faster to compute than per-class IoU
- B. classes with more pixels should count more
- C. it makes the score always higher than pixel accuracy
- D. it prevents a rare, poorly-segmented class from hiding behind common ones

<details>
<summary>Answer</summary>

**D. it prevents a rare, poorly-segmented class from hiding behind common ones** — Equal per-class weighting means a rare class you segment badly still tanks the mean — a feature. The cost: a tiny class's IoU can be volatile, so you always report per-class IoU too.

</details>

**9. Panoptic Quality factorizes as PQ = SQ × RQ, where SQ and RQ respectively capture:**

- A. semantic quality and regional quality of stuff only
- B. spatial resolution and receptive field
- C. how good the matched masks are (avg IoU of matches) and how many segments were correctly found without duplicates (an F1-like term)
- D. the softmax quality and the ReLU quality

<details>
<summary>Answer</summary>

**C. how good the matched masks are (avg IoU of matches) and how many segments were correctly found without duplicates (an F1-like term)** — SQ = mean IoU over matched (TP) segments; RQ = |TP|/(|TP|+½|FP|+½|FN|), an F1 on segment recognition. A model can have tight masks (high SQ) yet miss objects (low RQ) or vice versa.

</details>

**10. Pixel accuracy is a misleading segmentation metric primarily because:**

- A. it is mathematically equal to IoU
- B. it over-weights small objects
- C. it cannot be computed without ground-truth masks
- D. background/majority pixels dominate the count, so a model can score high while missing every small object

<details>
<summary>Answer</summary>

**D. background/majority pixels dominate the count, so a model can score high while missing every small object** — Weighting every pixel equally means the large stuff regions dominate; predicting the majority class everywhere scores high while missing small things — the accuracy trap at pixel scale.

</details>

**11. Soft-Dice loss survives severe foreground/background imbalance where plain cross-entropy fails because:**

- A. it ignores the background pixels entirely by masking them out
- B. it is computed only on boundary pixels
- C. the class fraction cancels in its ratio, making it scale-invariant to object size
- D. it has a larger learning rate

<details>
<summary>Answer</summary>

**C. the class fraction cancels in its ratio, making it scale-invariant to object size** — Dice's numerator and denominator both scale with region size, so a 100-pixel tumor and a huge organ contribute comparably; CE instead sums per pixel, so background dominates its gradient.

</details>

**12. Focal loss modifies cross-entropy by multiplying it by (1−p_t)^γ, which:**

- A. converts it into the Dice loss
- B. removes the logarithm from cross-entropy
- C. down-weights easy, well-classified pixels so gradient concentrates on hard/rare ones
- D. up-weights easy background pixels for stability

<details>
<summary>Answer</summary>

**C. down-weights easy, well-classified pixels so gradient concentrates on hard/rare ones** — For an easy pixel p_t→1 the factor →0, so easy examples contribute almost nothing and the loss focuses on hard, ambiguous pixels — 'soft hard-example mining' with no sampling heuristic.

</details>

**13. The Lovász-Softmax loss is principled as an IoU surrogate because it uses:**

- A. the Lovász convex extension of the Jaccard set function, which is submodular
- B. the exact non-differentiable IoU with a straight-through estimator
- C. a second-order Taylor expansion of pixel accuracy
- D. a random relaxation of the mask boundary

<details>
<summary>Answer</summary>

**A. the Lovász convex extension of the Jaccard set function, which is submodular** — The Jaccard loss over pixel errors is submodular; its tight piecewise-linear convex Lovász extension is differentiable and directly targets IoU (Berman et al., CVPR 2018).

</details>

**14. Query-based mask-classification models (MaskFormer/Mask2Former) unify semantic, instance, and panoptic segmentation by:**

- A. predicting a fixed set of queries, each emitting a class and a whole-image mask, matched to ground truth by bipartite matching
- B. detecting boxes first and then applying NMS to the masks
- C. using only atrous convolution at multiple rates
- D. running three separate CNN heads and averaging their outputs

<details>
<summary>Answer</summary>

**A. predicting a fixed set of queries, each emitting a class and a whole-image mask, matched to ground truth by bipartite matching** — Following DETR's set-prediction view, each query predicts (class, mask); the three tasks differ only in post-processing, and there is no box or NMS — objects are separated by which query claims them.

</details>

**15. In a high-stakes medical or driving deployment, the reason to report per-class (not just mean) IoU and evaluate on the long tail is that:**

- A. aggregate mIoU can be high while the model systematically fails on rare, safety-critical classes
- B. the mean is undefined when a class is rare
- C. per-class IoU is required to compute pixel accuracy
- D. it makes the score look better to reviewers

<details>
<summary>Answer</summary>

**A. aggregate mIoU can be high while the model systematically fails on rare, safety-critical classes** — A rare pedestrian/tumor class can be badly segmented while the mean stays high; safety requires cost-weighted, per-class, out-of-distribution evaluation plus calibration — not a single i.i.d. average.

</details>

---
