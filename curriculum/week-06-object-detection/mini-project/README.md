# Mini-Project — An Object Detection System with Honest COCO mAP

## Brief

Build a working object-detection system on images you care about, and evaluate it the way benchmarks do — with
correctly computed COCO-style mAP — proving you own boxes, IoU, NMS, label assignment, and honest detection
metrics. This is the deliverable that shows you can not just *run* a detector but *evaluate and diagnose* one.

## Requirements

1. **Detector.** Run a COCO-pretrained detector, or fine-tune one on a small custom class (transfer learning
   from Week 5). State the model, backbone, and weights.
2. **Core utilities.** Your own `iou` and `nms` (greedy), cross-checked against `torchvision.ops.nms`, actually
   used in your pipeline — not just defined. Include `giou` and show it has a gradient where IoU is flat.
3. **Inference.** Draw boxes with class labels and confidence scores on a set of your own images at a
   *justified* confidence threshold (show the detections-vs-threshold curve that justifies it).
4. **Ground truth.** Hand-label (or reuse a labeled subset of) enough images to evaluate — a few objects each,
   across at least two classes.
5. **Evaluation.** Per-class AP and mAP at **0.5, 0.75, and [.5:.95]**, computed with *your own* metric code
   (greedy IoU matching, one-GT-per-prediction, all-point interpolation), cross-checked against torchmetrics at
   IoU 0.5. Report TP/FP/FN counts and, if you have enough small objects, AP_S/AP_M/AP_L.
6. **Error analysis.** Name the failure modes — small objects, occlusion, class confusions, duplicate boxes,
   and ground-truth label errors — with example images and rough counts per mode.
7. **README.** Reproduction steps, the *exact* mAP definition you used (which interpolation, which thresholds),
   and honest limitations. State which mAP any headline number refers to.

## Stretch

- Fine-tune on a custom class and report **before/after** mAP@[.5:.95].
- Add the speed-accuracy frontier from Challenge 2 (mAP vs. latency for ≥3 models, Pareto-dominated ones
  marked).
- Add the one-to-one-vs-one-to-many with/without-NMS experiment from Challenge 3 on a small scale.

## What you are proving

You can deploy *and honestly evaluate and diagnose* an object detector — the most widely used vision capability
in industry — reporting a benchmark-grade metric you computed and can defend line by line, not a black-box
number. Next week you go finer still: not just a box around each object, but a pixel-perfect mask of its exact
shape.
