# Week 6 — Object detection

> **Goal:** by Sunday you can (1) represent and compare boxes precisely, deriving IoU and its differentiable relatives GIoU/DIoU and knowing exactly when each helps a regression loss, (2) implement greedy and Soft-NMS from scratch and state the optimization problem each approximates, (3) place any detector on the design map — two-stage, one-stage anchor-based, anchor-free, and set-prediction (DETR) — and explain the label-assignment problem that separates them, (4) derive the matched multi-part detection loss including focal loss from the class-imbalance analysis, and (5) compute COCO mAP@[.5:.95] correctly by hand — matching, PR-curve interpolation, area, and averaging — and read a per-class / per-scale AP breakdown honestly.

Classification answers *what is this image?*. Detection answers the harder, far more useful question — *what objects are present, and precisely **where**?* — emitting a labeled, scored bounding box for every instance. That single change of output type breaks almost every assumption from the classification weeks: the number of outputs is variable and unknown, the loss must jointly localize and classify, the label of a prediction is not given but must be **assigned** by matching to ground truth, and the background class outnumbers foreground by three or four orders of magnitude.

This week builds the full machinery. We start from **box geometry** and make **IoU** — the universal ruler of detection — rigorous, then extend it to the **GIoU/DIoU/CIoU** family (Rezatofighi et al., CVPR 2019; Zheng et al., AAAI 2020) that repairs IoU's zero-gradient blind spot so it can drive a regression loss. We build **non-maximum suppression** by hand — the same greedy idea you met thinning edges in Canny (Week 2) — and its softer successor **Soft-NMS** (Bodla et al., ICCV 2017), stating the objective each approximates. We map the **architecture space**: two-stage propose-then-classify (Faster R-CNN; Ren et al., NeurIPS 2015), one-stage dense prediction (YOLO, SSD, RetinaNet), anchor-free center/point prediction (FCOS; Tian et al., ICCV 2019), and end-to-end **set prediction** (DETR; Carion et al., ECCV 2020) that deletes anchors *and* NMS with a Hungarian bipartite-matching loss. Underneath all of them sits one problem — **label assignment**, deciding which predictions are responsible for which objects — and one imbalance fix, **focal loss** (Lin et al., ICCV 2017), which we derive. Finally we make evaluation exact: **COCO mAP@[.5:.95]**, computed by hand, read per-class and per-scale, never quoted without stating which mAP you mean. You will run a real detector, evaluate it honestly, and understand every number.

The engineering thesis of the week: a detector is not a magic box that emits boxes. It is a **dense set of candidate predictions plus an assignment rule plus a suppression rule plus a matched loss plus a benchmark protocol** — and every design in the literature is a different choice of those five parts.

## Learning objectives

By the end of this week, you will be able to:

- **Represent** detections as variable-length (box, label, score) sets, convert fluently between corner and center encodings, and compute IoU by hand with correct edge-case handling.
- **Derive** the GIoU/DIoU/CIoU family and explain precisely why plain IoU is a poor *loss* (zero gradient for disjoint boxes) even though it is the right *metric*.
- **Implement** greedy NMS and Soft-NMS from scratch, per class, and state the objective each one approximately solves.
- **Classify** any detector on the two-axis design map (stage count × anchor policy), and add the set-prediction paradigm, explaining the label-assignment problem that distinguishes them.
- **Derive** the matched two-part detection loss and focal loss FL(p_t) = -(1-p_t)^gamma log p_t from the foreground/background imbalance argument.
- **Explain** DETR's Hungarian bipartite matching and why end-to-end set prediction removes both hand-designed anchors and NMS.
- **Compute** COCO mAP@[.5:.95] end to end by hand — greedy IoU matching, precision-recall curve, all-point interpolation, area, and averaging over classes and IoU thresholds.
- **Diagnose** a detector from per-class and per-scale AP, the mAP@0.5-vs-0.75 gap, and a named-failure-mode error analysis rather than a single headline number.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CAP 4410` — localize objects in an image: box representation, IoU, non-maximum suppression, detector architectures, and mean Average Precision. |
| Industry | Report a detector's mAP the way the benchmark defines it, and defend the number line by line when somebody disputes it. |
| Beyond the bar | derives focal loss and the label-assignment problem, then carries on past both into set prediction that needs neither anchors nor NMS — `lecture-notes/05-detr-set-prediction.md` |

## Prerequisites

- Week 5 mini-project (transfer learning) committed and working — you will fine-tune a pretrained detector.
- Comfort with tensors, broadcasting, and a training loop (Weeks 1, 3, 4).
- Basic probability and the precision/recall vocabulary; the sigmoid, softmax, and cross-entropy (Week 4).
- The Canny non-maximum-suppression idea from Week 2 — NMS on boxes is the same principle.

## This week

**Lectures**

1. [Lecture 1 — Boxes, the IoU metric family, and why detection is hard](lecture-notes/01-boxes-and-iou.md)
2. [Lecture 2 — Non-max suppression (greedy and soft) & the detector design map](lecture-notes/02-nms-and-architectures.md)
3. [Lecture 3 — Training detectors, and computing COCO mAP correctly by hand](lecture-notes/03-training-and-map.md)
4. [Lecture 4 — Label assignment and focal loss: the theory that makes one-stage detection work](lecture-notes/04-label-assignment-and-focal-loss.md)
5. [Lecture 5 — Set prediction: DETR, Hungarian matching, and detection without anchors or NMS](lecture-notes/05-detr-set-prediction.md)

**Exercises**

1. [Exercise 1 — IoU, the GIoU family, and NMS (greedy and soft) from scratch](exercises/exercise-01-iou-and-nms.md)
2. [Exercise 2 — Run a pretrained detector and dissect its raw output](exercises/exercise-02-run-a-detector.md)
3. [Exercise 3 — Compute COCO-style mAP by hand and validate it against torchmetrics](exercises/exercise-03-measure-map.md)
4. [Exercise 4 — Label assignment and focal loss, quantified](exercises/exercise-04-assignment-and-focal-loss.md)

**Challenges**

1. [Challenge 1 — Fine-tune a detector on a custom class and analyze its failures](challenges/challenge-01-finetune-a-detector.md)
2. [Challenge 2 — Map the speed-accuracy frontier and pick on it](challenges/challenge-02-speed-accuracy.md)
3. [Challenge 3 — Build a one-to-one matched loss and observe NMS becoming unnecessary (open)](challenges/challenge-03-nms-free-set-loss.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 7.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook that runs a COCO-pretrained detector on your own images, draws predicted boxes with labels and scores, applies your own hand-written IoU + greedy-NMS (cross-checked against torchvision.ops.nms) in the pipeline, and computes mAP against a small hand-labeled ground-truth set — reported at IoU 0.5 and 0.75, per class, with a named-failure-mode error analysis and a one-paragraph account of which mAP definition you used and why the 0.5-vs-0.75 gap is what it is.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
