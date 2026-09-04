# Week 6 — Graduate Problem Set: Boxes, IoU, NMS, Assignment, Focal Loss, and mAP

Ten problems, easy to hard, mixing computation, derivation, proof, and open analysis. Solution
sketches follow — attempt each fully before reading them. Notation: boxes in corner form `(x1,y1,x2,y2)`;
`p_t` is the model's probability of the correct class; AP is average precision (area under the interpolated
precision-recall curve).

**P1 (IoU by hand).** Boxes A = (0,0,4,4) and B = (2,2,6,6). Compute the intersection area, union area, and
IoU. Then move B to (5,5,9,9) and recompute — what is the IoU, and what does its being constant for all
positions of a disjoint B imply for a `1 − IoU` loss?

**P2 (box-format conversion).** A box is given in center form `(cx, cy, w, h) = (10, 8, 6, 4)`. Give its corner
form. A regression head predicts `(Δcx, Δcy, Δlog w, Δlog h)`; explain why width/height are regressed in log
space rather than linearly.

**P3 (GIoU gradient).** Define GIoU = IoU − |C \ (A ∪ B)| / |C|, where C is the smallest axis-aligned box
enclosing A and B. For two equal 2×2 boxes separated horizontally by a gap `g > 0` (so IoU = 0), write GIoU as
a function of `g` and show `dGIoU/dg < 0` — i.e. GIoU rewards closing the gap where IoU gives no signal.

**P4 (NMS trace).** Four boxes with scores [0.9, 0.8, 0.7, 0.6] and pairwise IoUs: (1,2)=0.6, (1,3)=0.1,
(2,3)=0.05, (3,4)=0.7, others small. Trace greedy NMS at threshold 0.5: list the kept set in order. Then trace
Soft-NMS with linear decay `s ← s(1 − IoU)` for the first suppression step and state how the ranking changes.

**P5 (per-class AP).** For one class with 3 ground truths, predictions sorted by score give the TP/FP sequence
[TP, FP, TP, TP, FP] (matched at IoU ≥ 0.5). Build the precision and recall after each prediction, then compute
AP using all-point interpolation (integrate the monotonized precision envelope). Show your PR table.

**P6 (why one GT per prediction).** Prove that if the matching rule allowed a single ground truth to be matched
by *multiple* predictions, a detector could inflate its recall to 1 trivially. Explain how the one-GT-per-
prediction rule makes NMS "count" in the metric.

**P7 (focal loss algebra).** For binary cross-entropy CE(p_t) = −log p_t and focal loss FL(p_t) =
−(1−p_t)^γ log p_t, compute the ratio FL/CE at p_t ∈ {0.1, 0.5, 0.9} for γ = 2. Interpret each number as a
down-weighting factor and identify at which p_t the suppression is strongest in absolute terms.

**P8 (imbalance inversion).** An image has `F` foreground examples at p_t = 0.3 and `B` background at p_t =
0.95. Write the total CE loss and the total FL (γ = 2) loss as functions of F and B. Find the ratio B/F at
which background stops dominating under FL but still dominates under CE, quantifying the inversion.

**P9 (mAP protocol proof).** Show that for a fixed set of predictions and ground truths, mAP@[.5:.95] ≤
mAP@0.5. Under what precise condition on the detector's localization does the *gap* mAP@0.5 − mAP@0.75 become
large? Tie your answer to a diagnostic conclusion about the detector.

**P10 (open — Hungarian set loss).** DETR trains with a one-to-one bipartite matching found by the Hungarian
algorithm. Argue, from the structure of the training gradient, why this makes NMS unnecessary, and explain why
a naive one-to-many assignment does not. Then discuss one reason the original DETR converged ~10× slower than
Faster R-CNN and how a successor (Deformable DETR or DINO) addressed it. (Open-ended; argue carefully.)

---

## Solution sketches

**S1.** Intersection (2,2,4,4) → 2×2 = 4; areas 16 and 16; union = 16+16−4 = 28; IoU = 4/28 ≈ 0.143. Disjoint B:
IoU = 0, and it stays 0 for every disjoint position — so `1−IoU` has zero gradient and cannot localize a
non-overlapping box; this motivates GIoU (P3).

**S2.** Corner form: `x1 = 10−3 = 7, y1 = 8−2 = 6, x2 = 13, y2 = 10` → (7,6,13,10). Log-space size regression
keeps predictions scale-invariant and positive (a box cannot have negative width), and makes equal *relative*
size errors cost equally — the standard Faster R-CNN encoding.

**S3.** With gap `g`, enclosing box width = 2+g+2 = 4+g (height 2), so |C| = 2(4+g); A∪B area = 8 (disjoint,
IoU 0); |C \ (A∪B)| = 2(4+g) − 8 = 2g; GIoU = 0 − 2g/(2(4+g)) = −g/(4+g). Then dGIoU/dg = −4/(4+g)² < 0, so GIoU
increases (toward 0) as g shrinks — a usable gradient where IoU has none.

**S4.** Greedy: keep 1 (0.9); box 2 has IoU 0.6 > 0.5 → suppressed; keep 3 (0.7, IoU with 1 = 0.1); box 4 has
IoU 0.7 with box 3 → suppressed. Kept = {1, 3}. Soft-NMS first step: box 2's score → 0.8(1−0.6) = 0.32
(demoted, not deleted), so it survives to be re-ranked below box 3 and may be kept at low confidence — trading a
possible duplicate for recall in a crowd.

**S5.** Cumulative TP/FP: after preds → (1,0),(1,1),(2,1),(3,1),(3,2). Precision = 1, 0.5, 0.667, 0.75, 0.6;
recall = 1/3, 1/3, 2/3, 1, 1. Monotonize precision from the right: envelope = 0.75 up to recall 1. All-point AP
≈ 0.75 (the max precision achievable at every recall level is 0.75, sustained to recall 1). Show the table
explicitly.

**S6.** If one GT could satisfy many predictions, stack all N predictions on the single GT: every one is a TP,
recall → 1, precision high — trivially perfect without detecting the other objects. Forbidding it makes only the
*first* (highest-scoring) prediction a TP and the rest duplicate FPs, so a model that fails to suppress
duplicates is penalized — which is precisely why NMS matters to the score.

**S7.** (1−p_t)² at p_t = 0.1/0.5/0.9 = 0.81 / 0.25 / 0.01, so FL/CE = 0.81, 0.25, 0.01. Easy examples (p_t=0.9)
are suppressed ~100×; the *relative* suppression is strongest at high p_t. In absolute loss, the largest removed
quantity is on the many easy examples, which is the point.

**S8.** CE total = F·(−log0.3) + B·(−log0.95) ≈ 1.204F + 0.0513B. FL total = F·0.49·1.204 + B·0.0025·0.0513 ≈
0.590F + 0.000128B. Background dominates CE when 0.0513B > 1.204F, i.e. B/F > 23.5; under FL background dominates
only when 0.000128B > 0.590F, i.e. B/F > ~4600. So for any 23.5 < B/F < 4600, CE is background-dominated while FL
is foreground-dominated — the inversion, quantified.

**S9.** AP at a stricter IoU threshold counts fewer predictions as TP (some IoU-0.5 matches fall below 0.75), so
the PR curve can only shift down or stay — hence AP(0.75) ≤ AP(0.5) pointwise, and averaging 0.5:0.95 (which
includes thresholds ≥ 0.5) gives mAP@[.5:.95] ≤ mAP@0.5. The 0.5−0.75 gap is large exactly when many detections
overlap their GT loosely (0.5 ≤ IoU < 0.75): the detector *finds* objects but localizes imprecisely — a
box-regression/anchor-scale problem, not a recognition one.

**S10.** One-to-one matching assigns each GT to a single prediction and pushes all others to ∅; the gradient
therefore *penalizes* a second prediction firing on an object, so the network learns not to duplicate — no NMS
needed. One-to-many assigns many positives per object, *rewarding* redundant firing, so duplicates must be
removed post hoc by NMS. Original DETR's slow convergence came from unstable early matching and hard-to-learn
global cross-attention over dense features; Deformable DETR replaced it with sparse multi-scale deformable
attention (~10× faster, better small objects), and DINO added query denoising and contrastive matching for
stability and SOTA AP.
