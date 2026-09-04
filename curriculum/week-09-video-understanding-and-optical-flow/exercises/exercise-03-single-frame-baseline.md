# Exercise 3 — The single-frame baseline (the truth serum)

**Goal:** measure whether temporal modeling actually earns its cost.

## Tasks

1. On a small labeled action-clip set, build a **single-frame baseline**: run an image classifier
   (or the video model's spatial features) on one — or a few averaged — frame(s) per clip and predict
   the action, aggregating if needed.
2. Compare its accuracy to the temporal model from Exercise 2 **on the same clips**, using a per-action
   confusion breakdown, not just top-1 accuracy.
3. Identify which actions the single-frame baseline gets right (appearance-defined, visible in one
   frame) and which need temporal context (motion-defined).
4. Estimate the **compute cost** difference: FLOPs or wall-clock inference latency per clip, and peak
   memory, for the two approaches.
5. Ensure the evaluation splits **by video**, not by frame — state explicitly how you avoided leakage.

## Deliverable

A notebook comparing single-frame vs. temporal accuracy *per action*, with a note on which actions
truly needed time, the measured cost difference, and a verdict on whether the temporal model's cost was
justified. State your video-level split. This baseline is the honesty check for all video work.
