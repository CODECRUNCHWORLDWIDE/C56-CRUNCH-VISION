# Challenge 2 — Stress-test a segmenter, choose the loss, and reason about the ceiling

Segmentation fails in specific, diagnosable ways. Find them, match the loss to the task, and confront the
irreducible limits.

1. **Failure gallery.** Run a segmenter on hard cases: heavy occlusion, unusual lighting, tiny objects, reflective/
   transparent surfaces, and classes near the edge of its training distribution. Catalog each failure mode with images
   and name its likely cause (receptive field, boundary ambiguity, out-of-distribution, scale, imbalance).
2. **Loss choice, measured.** On an imbalanced task (object is a small fraction of pixels), train with pixel-wise CE vs.
   Dice loss vs. CE+Dice. Report **foreground** IoU/Dice (not aggregate accuracy) and show which best segments the small
   object; explain via the gradient-domination argument (CE's gradient is dominated by background pixels).
3. **The ground-truth ceiling.** Have two people (or two annotation attempts) label the same object's boundary in 3
   images and compute the **inter-annotator IoU**. Discuss what it means that this number is below 1.0: it *caps* the
   achievable mIoU, so a model scoring near it is not "wrong" — it is at the noise floor. Relate this to Lecture 5's
   point that in medicine the ground truth is a distribution (STAPLE), not a single truth.
4. **Calibration probe (stretch).** Bin the model's predicted foreground probabilities and plot a reliability diagram;
   report whether it is over- or under-confident, and why that matters for a downstream decision.

**Deliverable:** a failure-mode gallery with named causes, the CE-vs-Dice-vs-combined comparison with foreground
metrics and explanation, an inter-annotator IoU measurement with a paragraph on the achievable-score ceiling, and
(stretch) a reliability diagram. Understanding *why* and *where* segmentation breaks, matching the loss to the task,
and knowing the ceiling is professional judgment, not benchmark chasing.
