# Challenge 3 — Panoptic segmentation and decomposing PQ into SQ × RQ

Go beyond semantic and instance to the unified task, and learn to read the metric that governs it.

1. Run a pretrained **panoptic** model (Mask2Former panoptic, or `detectron2`'s Panoptic FPN) on a handful of complex
   multi-object scenes. Visualize the panoptic output as a single gap-free, non-overlapping colored partition (things
   get per-instance colors, stuff gets per-class colors).
2. Implement **Panoptic Quality** from Lecture 3's definition: for one image (or a tiny labeled set you annotate),
   match predicted to ground-truth segments by the IoU>0.5 rule, then compute `PQ = ΣIoU / (|TP| + ½|FP| + ½|FN|)` and its
   factorization `PQ = SQ × RQ`. Validate on a hand-computed toy case (a couple of segments with known overlaps).
3. **Read the decomposition.** Construct or find one scene where SQ is high but RQ is low (good masks, but the model
   misses or duplicates objects) and one where the reverse holds. Show numerically how PQ responds, and explain what each
   sub-metric is telling you that a single mIoU number could not.
4. **Contrast with instance mask-AP.** Briefly explain why panoptic needed PQ rather than reusing mask-AP + mIoU — what
   does the non-overlap partition constraint add?

**Deliverable:** panoptic overlays on complex scenes, a tested `panoptic_quality` implementation validated on a toy
case, two contrasting scenes demonstrating the SQ-vs-RQ decomposition with numbers, and a paragraph on why the unified
task demanded a unified metric. This is the state-of-the-art evaluation, done by hand so you actually understand it.
