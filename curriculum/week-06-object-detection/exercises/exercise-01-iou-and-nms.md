# Exercise 1 — IoU, the GIoU family, and NMS (greedy and soft) from scratch

**Goal:** build detection's core geometric utilities by hand and understand why the loss variants
exist.

## Tasks

1. Implement `iou(box_a, box_b)` in corner form with clamped intersection. Test on hand-computed cases:
   identical boxes (IoU = 1), disjoint (0), and the (0,0,2,2)/(1,1,3,3) partial overlap from Lecture 1
   (IoU = 1/7). Assert each against your paper value.
2. Implement `giou(box_a, box_b)` = IoU − (area of smallest enclosing box not covered by the union) / (area of
   enclosing box). Show that for two **disjoint** boxes GIoU is strictly negative and *increases* as you slide
   them closer — i.e. it has a gradient where IoU is flat 0. Plot GIoU vs. horizontal offset for a pair of
   equal boxes; overlay IoU to show its dead zone.
3. Implement greedy `nms(boxes, scores, iou_thresh)` from Lecture 2. Verify it against `torchvision.ops.nms`
   with `torch.allclose` on the kept indices for a random pile of 50 boxes.
4. Implement `soft_nms(boxes, scores, sigma)` with Gaussian score decay `s_j *= exp(-iou² / sigma)`. Construct
   two heavily overlapping *real* objects (IoU ≈ 0.6) plus duplicates; show greedy NMS deletes one real object
   while Soft-NMS retains both at lower rank.
5. Sweep the greedy IoU threshold (0.3, 0.5, 0.7) on a synthetic pile of ~10 boxes around two objects and show
   how it changes what survives — including a too-low case that wrongly merges two adjacent objects.

## Deliverable

A notebook with tested `iou`, `giou`, `nms`, and `soft_nms`; the GIoU-vs-offset plot demonstrating IoU's dead
zone; the `allclose` check against `torchvision.ops.nms`; and the greedy-vs-soft crowded-object comparison with
before/after box visualizations. These functions underpin every metric and loss this week.
