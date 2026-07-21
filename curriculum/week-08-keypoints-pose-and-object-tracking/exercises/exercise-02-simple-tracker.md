# Exercise 2 — Build an IoU-based tracker

**Goal:** the detect-then-associate loop with stable IDs.

## Tasks

1. On a short clip (a few seconds; split into frames), run your object detector per frame.
2. Implement a simple tracker: for each new frame, match detections to existing tracks by IoU (assign each detection to the highest-IoU track above a threshold); create new tracks for unmatched detections; delete tracks unseen for N frames.
3. Assign and draw a persistent ID (and color) per track across the whole clip.
4. Play back the annotated frames and check that IDs stay consistent for objects that move steadily.

## Deliverable

An annotated clip (or frame sequence) where each object keeps a consistent ID number, plus your tracker code. This is SORT's core, hand-built.
