# Exercise 1 — Implement IoU and non-max suppression

**Goal:** build detection's two core utilities by hand.

## Tasks

1. Implement `iou(box_a, box_b)` in corner form. Test it on hand-computed cases: identical boxes (IoU=1), non-overlapping (0), and a known partial overlap you compute on paper.
2. Implement greedy `nms(boxes, scores, iou_thresh)` from Lecture 2.
3. Create a synthetic pile of ~10 overlapping boxes around two "objects" with varied scores. Run your NMS and confirm it keeps one box per object.
4. Sweep the IoU threshold (0.3, 0.5, 0.7) and show how it changes what survives — including a case where too-low a threshold wrongly merges two adjacent objects.

## Deliverable

A notebook with tested `iou` and `nms`, a before/after box visualization, and the threshold sweep. Verify against `torchvision.ops.nms`. These functions underpin every metric this week.
