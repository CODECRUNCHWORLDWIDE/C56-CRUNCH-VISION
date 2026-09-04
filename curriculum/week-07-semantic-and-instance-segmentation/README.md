# Week 7 — Semantic & instance segmentation

> **Goal:** by Sunday you can (1) state segmentation as structured per-pixel prediction and explain the receptive-field-versus-resolution tension that every architecture is built to resolve, deriving why fully-convolutional nets, U-Net skips, and DeepLab's atrous/ASPP each trade differently; (2) explain the shift from 'detect-then-mask' (Mask R-CNN, with RoIAlign's sub-pixel argument) to unified query-based mask classification (DETR → Mask2Former) and why panoptic quality became one coherent metric; (3) derive IoU, Dice, and Panoptic Quality from set theory, prove Dice = 2·IoU/(1+IoU), and reason precisely about why pixel accuracy, cross-entropy, and even mIoU can each mislead; and (4) choose a segmentation loss (weighted CE, focal, soft-Dice, Lovász-Softmax, boundary) from the class imbalance and the metric you are optimizing — then run, evaluate, and honestly critique a segmenter on your own images.

Detection (Week 6) draws a box; **segmentation** labels every pixel. That extra precision is not a cosmetic upgrade — it changes the mathematics of the problem. A classifier emits one label; a detector emits a handful of boxes; a segmenter emits a **structured object**: an H×W label field whose pixels are statistically coupled (neighbors share labels, boundaries are smooth) and whose error is dominated by a vanishingly thin set of boundary pixels. Getting this right is what lets a surgical model trace a tumor's exact margin, a self-driving stack separate drivable road from curb, and a mapping system measure how much of a watershed is impervious surface. This week we treat dense prediction at graduate depth.

The usual course names three tasks (semantic, instance, panoptic) and three networks (U-Net, Mask R-CNN, DeepLab) and stops. We will not. You will see *why* a plain classification backbone is structurally unable to produce sharp masks — its 32× downsampling throws away exactly the spatial detail a boundary needs — and the three distinct escapes the field invented: **learned upsampling with skips** (FCN, U-Net; Long et al., CVPR 2015; Ronneberger et al., MICCAI 2015), **atrous convolution to enlarge the receptive field without losing resolution** (DeepLab; Chen et al., TPAMI 2018), and **global attention that has full receptive field from layer one** (SETR/SegFormer/Mask2Former; Zheng et al., CVPR 2021; Xie et al., NeurIPS 2021; Cheng et al., CVPR 2022). On instance segmentation you will follow the arc from Mask R-CNN's RoIAlign — whose whole point is a sub-pixel bilinear argument that fixes RoIPool's quantization — to query-based models that dissolve the box entirely and predict masks as set prediction.

Throughout, the metric is the discipline. You will derive IoU and Dice as functions on sets, prove their monotone relationship, understand why mIoU equal-weights classes on purpose, meet **Panoptic Quality** as the clean product of recognition and segmentation quality (Kirillov et al., CVPR 2019), and confront the uncomfortable truth that human annotators disagree at boundaries — so there is an inherent ceiling on any score, and a number without an overlaid picture is not an evaluation. For the high-stakes domains — medicine, remote sensing, autonomy — we treat calibration, label noise, and distribution shift as first-class safety concerns, not footnotes.

## Learning objectives

By the end of this week, you will be able to:

- **Formulate** segmentation as structured per-pixel prediction and articulate the receptive-field-versus-resolution tension, deriving each pixel's theoretical receptive field through a downsampling stack.
- **Compare** the three architectural escapes — encoder-decoder + skips (FCN/U-Net), atrous convolution + ASPP (DeepLab), and attention-based segmenters (SETR/SegFormer/Mask2Former) — by what each trades in FLOPs, receptive field, and boundary fidelity.
- **Explain** the shift from detect-then-mask instance segmentation (Mask R-CNN, RoIAlign's bilinear sub-pixel sampling) to unified mask classification (Mask2Former), and why panoptic segmentation unified the two tasks.
- **Derive** IoU, Dice, and Panoptic Quality from set operations, prove Dice = 2·IoU/(1+IoU), and state precisely why pixel accuracy, mIoU, and boundary-blind metrics each mislead in different regimes.
- **Select and justify** a training loss — weighted cross-entropy, focal, soft-Dice, Lovász-Softmax, or a boundary loss — from the class imbalance and the evaluation metric, and explain the differentiability trick each depends on.
- **Run and evaluate** pretrained semantic and instance/panoptic segmenters on your own images, overlaying masks and computing per-class IoU/Dice against ground truth you create.
- **Diagnose** segmentation failure visually and statistically — leaky boundaries, merged instances, missed thin structures, class confusion — and connect each to an architectural or loss cause.
- **Assess** calibration, label noise, and distribution shift for a high-stakes segmentation deployment, and state the honest limitations (annotation ambiguity, ceiling on achievable score) in a model card.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 4670` — produce dense per-pixel predictions, semantic and instance, and measure them with region-overlap metrics rather than pixel accuracy. |
| Industry | Measure a segmenter against masks you drew yourself, and say out loud where the annotation is ambiguous enough to cap any score the model could reach. |
| Beyond the bar | proves Dice = 2·IoU/(1+IoU) and then runs a loss lab that picks the training loss from the class imbalance and the evaluation metric — `exercises/exercise-04-loss-lab.md` |

## Prerequisites

- Week 6's detector project committed and working — you reuse IoU, NMS, and the anchor/RoI machinery here.
- Comfort with CNN backbones, downsampling/stride, and receptive fields (Weeks 3–5).
- PyTorch: writing a training loop, autograd, and using torchvision pretrained models.
- Basic set theory and probability (for the metric derivations) and the softmax/cross-entropy from earlier weeks.

## This week

**Lectures**

1. [Lecture 1 — Semantic, instance & panoptic: dense prediction as a structured problem](lecture-notes/01-kinds-of-segmentation.md)
2. [Lecture 2 — The resolution tension: FCN, U-Net, DeepLab, and Mask R-CNN](lecture-notes/02-encoder-decoder-architectures.md)
3. [Lecture 3 — Measuring masks honestly: IoU, Dice, PQ, boundaries, and calibration](lecture-notes/03-segmentation-metrics.md)
4. [Lecture 4 — The loss zoo for dense prediction: from weighted cross-entropy to Lovász-Softmax](lecture-notes/04-losses-for-dense-prediction.md)
5. [Lecture 5 — Query-based segmenters and segmentation in high-stakes domains](lecture-notes/05-transformers-and-high-stakes-segmentation.md)

**Exercises**

1. [Exercise 1 — Run a semantic segmenter and read the label field](exercises/exercise-01-run-semantic-seg.md)
2. [Exercise 2 — Implement IoU, Dice, and mIoU, and expose the accuracy trap](exercises/exercise-02-iou-dice.md)
3. [Exercise 3 — Instance & panoptic segmentation: separating objects](exercises/exercise-03-instance-seg.md)
4. [Exercise 4 — Loss lab: cross-entropy vs. Dice vs. focal on an imbalanced target](exercises/exercise-04-loss-lab.md)

**Challenges**

1. [Challenge 1 — Train a U-Net and prove skip connections earn their keep](challenges/challenge-01-train-a-unet.md)
2. [Challenge 2 — Stress-test a segmenter, choose the loss, and reason about the ceiling](challenges/challenge-02-segmentation-in-the-wild.md)
3. [Challenge 3 — Panoptic segmentation and decomposing PQ into SQ × RQ](challenges/challenge-03-panoptic-and-pq.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 8.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook + short report that: runs a pretrained semantic segmenter (DeepLabV3 / SegFormer) and an instance/panoptic segmenter (Mask R-CNN / Mask2Former) on your own images; overlays per-class and per-instance masks; implements `mask_iou`, `dice`, and a mIoU aggregator from scratch and validates them against a reference; computes per-class IoU/Dice against several hand-drawn ground-truth masks (never pixel accuracy alone); and delivers an honest error gallery plus a paragraph on boundary-label ambiguity and the resulting ceiling on any achievable score.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
