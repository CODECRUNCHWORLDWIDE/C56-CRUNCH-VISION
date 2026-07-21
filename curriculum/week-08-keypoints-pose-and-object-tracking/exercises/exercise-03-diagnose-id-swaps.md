# Exercise 3 — Find and reduce identity switches

**Goal:** see tracking's signature failure and mitigate it.

## Tasks

1. Run your Exercise 2 tracker on a clip where two same-class objects **cross or occlude** each other. Watch for ID swaps at the crossing.
2. Count the identity switches by eye (or against a few hand-labeled frames). Confirm IoU-only association swaps IDs when boxes overlap.
3. Add a crude appearance cue: for each object, compute a simple embedding (a color histogram, or features from a pretrained backbone) and use it to break ties when IoU is ambiguous — the Deep SORT idea, simplified.
4. Show the appearance cue reduces ID swaps at the crossing.

## Deliverable

A before/after comparison of ID swaps at a crossing, with and without the appearance cue, and a note on why position alone isn't enough. This is exactly the problem Deep SORT solved.
