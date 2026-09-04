# Week 6 — Homework

Cement detection's vocabulary, losses, and metrics before segmentation. These take a few focused hours and set up Week 7's move from a box around each object to a pixel-perfect mask of its exact shape. Do the IoU and focal-loss derivations by hand before coding — the geometry and the imbalance argument must be instinct before a library hides them.

## Tasks

- Compute IoU by hand for two boxes you define (with a worked intersection/union), then verify against your code; repeat for a disjoint pair and state why `1 − IoU` gives no gradient there.
- Write, in your own words, the greedy NMS algorithm, why it runs per class, and how Soft-NMS changes the suppression step — and give one scene where Soft-NMS keeps a box greedy NMS would wrongly delete.
- Derive focal loss from binary cross-entropy: state the modulating factor, and compute the down-weighting at p_t = 0.9 and p_t = 0.1 for γ = 2, explaining the imbalance-inverting effect.
- Explain the difference between mAP@0.5 and mAP@[.5:.95], why they are not comparable, and what a large mAP@0.5-minus-mAP@0.75 gap tells you diagnostically about a detector.
- Read the torchvision detection finetuning tutorial and the DETR paper's method section; summarize in a paragraph how DETR removes both anchors and NMS via one-to-one Hungarian matching.
- Extend your mini-project to report per-class AP and TP/FP/FN counts at IoU 0.5 and 0.75, and add a two-sentence error-analysis note naming your detector's dominant failure mode with one example image.

## Definition of done

A committed notebook and README that: run (or fine-tune) a COCO-pretrained detector on your own images; use your own IoU + greedy-NMS (cross-checked against torchvision) in the pipeline; draw boxes with labels and scores at a justified confidence threshold; and compute per-class AP and mAP at 0.5, 0.75, and [.5:.95] with your own metric code (agreeing with torchmetrics at 0.5), including TP/FP/FN counts, a named-failure-mode error analysis with example images, and an explicit statement of which mAP definition each reported number uses.

Submit by committing your work to your course repo under `week-06/`.
