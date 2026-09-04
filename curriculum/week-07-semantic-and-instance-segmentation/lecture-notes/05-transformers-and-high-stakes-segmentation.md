# Lecture 5 — Query-based segmenters and segmentation in high-stakes domains

Two frontiers close this week. First, the architectural state of the art has moved from convolutional
encoder–decoders and detect-then-mask pipelines to **query-based mask classification** built on transformers — one
formulation that unifies semantic, instance, and panoptic segmentation. Second, most segmentation that *matters* is
deployed in high-stakes domains — medicine, remote sensing, autonomous driving — where evaluation, calibration, label
noise, and distribution shift are safety questions, not benchmark footnotes. This lecture treats both.

## From per-pixel labeling to mask classification

The convolutional view treats semantic segmentation as per-pixel classification and instance segmentation as
detect-then-mask — two different formulations needing two different systems, awkwardly stitched for panoptic. The
transformer view (following DETR; Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020) reframes
*everything* as **set prediction**: predict a fixed set of `N` "queries," each of which emits (a) a class and (b) a
binary mask over the whole image. Semantic, instance, and panoptic then differ only in post-processing, not
architecture. DETR removed anchors and NMS by using bipartite (Hungarian) matching between predictions and ground truth
and a set-based loss — no hand-designed spatial priors, no duplicate-suppression heuristic.

- **SETR** (Zheng et al., CVPR 2021) first showed a pure Vision-Transformer encoder (Week 5's ViT) could do semantic
  segmentation, replacing the convolutional encoder with global self-attention that has **full receptive field from the
  first layer** — dissolving the receptive-field-vs-resolution tension of Lecture 2 at a quadratic compute cost.
- **SegFormer** (Xie et al., NeurIPS 2021) made this efficient: a hierarchical transformer encoder producing multi-scale
  features plus an all-MLP decoder, hitting strong mIoU with far fewer FLOPs and no positional-encoding fragility.
- **MaskFormer / Mask2Former** (Cheng et al., NeurIPS 2021 / CVPR 2022) delivered the unification: a query-based model
  where each query predicts a class and a mask, trained with mask classification, that achieves state of the art on
  **all three** tasks with **one architecture**. Mask2Former's masked-attention (attending only within predicted mask
  regions) both speeds convergence and sharpens masks.

The conceptual payoff: instance segmentation no longer needs a box at all — the mask is predicted directly and objects
are separated by *which query* claims them, not by NMS over boxes. And panoptic stops being two systems glued together.

## The other frontier: promptable and foundation segmenters

**Segment Anything** (Kirillov et al., *SAM*, ICCV 2023) trained a *promptable* segmenter on a billion masks: given a
point, box, or text-adjacent prompt, it returns a mask for essentially any object, zero-shot. It reframes segmentation
as a foundation-model capability — powerful for annotation acceleration and interactive tools, but with the usual
foundation-model caveats: no class semantics by default, uncertain calibration, and failure modes that are hard to
predict. Treat it as a strong prior/annotation aid, not a validated domain model.

## High-stakes domain 1 — medical imaging

Here the *loss* (Dice, boundary), the *metric* (Dice, Hausdorff distance for margins), and the *ethics* all sharpen. A
segmentation of a tumor or organ-at-risk feeds directly into radiotherapy planning or surgical decisions, so: **(a)
label noise is real and unavoidable** — expert radiologists disagree; the ground truth is a distribution, and one should
train and evaluate against multiple annotators where possible (the STAPLE framework, Warfield et al., IEEE TMI 2004,
fuses multiple expert segmentations into a probabilistic consensus). **(b) Calibration is a patient-safety property** —
an overconfident boundary can under-dose a tumor edge. **(c) Distribution shift is lethal to trust** — a model trained
on one scanner/protocol/hospital routinely degrades on another (domain shift across sites), so external validation is
mandatory, not optional. **(d) The model is decision-support, not the decision** — a human clinician must remain in the
loop, and the failure modes must be characterized in a model card. nnU-Net's dominance (Isensee et al., *Nature Methods*
2021) also carries a governance lesson: reproducible, well-validated pipelines beat novel architectures that are not
rigorously evaluated.

## High-stakes domain 2 — autonomous driving

Cityscapes-style semantic and panoptic segmentation feed a driving planner. The safety framing: **the long tail
dominates risk.** A model with excellent mIoU can still fail on the rare, decisive case — a pedestrian in unusual
clothing, a construction zone, a reflection. Aggregate mIoU hides this; you must evaluate on curated hard cases and rare
classes *separately* (report per-class IoU for pedestrian/cyclist, not just the mean), and treat under-segmentation of a
vulnerable road user as categorically worse than a stuff-class error — the metric should be *cost-weighted*, not
symmetric. Calibration and out-of-distribution detection (does the model *know* it is unsure at a strange object?) are
required for safe fallback behavior.

## High-stakes domain 3 — remote sensing

Land-cover and change detection from satellite/aerial imagery inform policy, agriculture, and disaster response. The
pitfalls are their own: **massive class imbalance** (rare land classes), **geographic distribution shift** (a model
trained on one continent's imagery generalizes poorly to another), **label noise** from coarse or outdated ground-truth
maps, and **spatial autocorrelation** that makes naive random train/test splits leak information across adjacent tiles —
you must split **geographically**, not randomly, or you will massively overstate accuracy. Ethically, segmentation of
human settlements and land use has surveillance and dual-use implications that belong in the project's risk assessment.

## The common thread

Across all three, the honest-evaluation discipline of Lecture 3 becomes a safety requirement: report per-class not just
mean; evaluate on the hard tail and out-of-distribution, not just the i.i.d. test set; measure calibration; account for
label noise and the annotator-disagreement ceiling; and write down limitations and intended use in a model card. The
architecture (U-Net, DeepLab, Mask2Former, SAM) is the easy part; trustworthy evaluation and deployment framing is the
graduate skill.

**Takeaway:** the state of the art reframes all segmentation as query-based *mask classification* — DETR's set
prediction, realized by SETR/SegFormer and unified across semantic/instance/panoptic by Mask2Former, with SAM pushing
segmentation toward a promptable foundation capability — dissolving both the receptive-field tension and the
detect-then-mask split. But most segmentation that matters is deployed in medicine, driving, and remote sensing, where
label noise, calibration, distribution shift (scanner, geography, the long tail), and cost-asymmetric errors make honest,
per-class, out-of-distribution evaluation and human-in-the-loop deployment a safety obligation, not a benchmark nicety.
