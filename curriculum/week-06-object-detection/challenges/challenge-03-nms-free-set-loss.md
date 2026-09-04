# Challenge 3 — Build a one-to-one matched loss and observe NMS becoming unnecessary (open)

This is an open, research-flavored challenge inspired by DETR (Lecture 5): demonstrate, on a small
scale, that a **one-to-one** matched loss trains a model that does not need NMS — the mechanism behind
end-to-end detection.

**Setup.** You do not need a full Transformer. Use a small dense detector (or even a toy grid predictor on
synthetic scenes of a few objects) that emits many candidate boxes. Train it two ways on identical data:

- **One-to-many (baseline):** the usual assignment where every anchor with IoU ≥ 0.5 to a GT is a positive.
  Many predictions fire per object; NMS is required at inference.
- **One-to-one (set loss):** compute the optimal bipartite assignment between predictions and ground truth with
  the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) using a matching cost that mixes class
  score and box (L1 + GIoU). Supervise exactly one prediction per GT; push all others toward "no object."

**Investigate.** For each trained model, at inference *without* NMS, count how many high-confidence predictions
fire per object. Show that the one-to-many model emits duplicates (so raw mAP without NMS is hurt by duplicate
false positives) while the one-to-one model emits ~one box per object (so its mAP barely changes when you skip
NMS). Report mAP with and without NMS for both.

**Analyze.** Explain, in terms of the training gradient, *why* the one-to-one loss discourages duplicates and
the one-to-many loss rewards them. Where does your toy result agree with DETR's claims, and where does the
small scale or lack of a Transformer break the analogy?

**Deliverable:** a short report with the with/without-NMS mAP table for both losses, the per-object
prediction-count comparison, and an honest analysis of the mechanism and the limits of your experiment. Negative
or partial results, well-analyzed, earn full credit — the graded skill is experimental rigor, not confirming a
headline.
