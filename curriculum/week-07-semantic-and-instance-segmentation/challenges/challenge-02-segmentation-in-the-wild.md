# Challenge 2 — Stress-test a segmenter and choose a loss

Segmentation fails in specific, diagnosable ways. Find them, and reason about the loss.

1. Run a segmenter on hard cases: heavy occlusion, unusual lighting, tiny objects, and classes near the edge of its training distribution. Catalog the failure modes with images.
2. On an imbalanced task (object is a small fraction of pixels), compare training with pixel-wise cross-entropy vs. Dice loss. Show which better segments the small object and explain why (cross-entropy is dominated by background pixels).
3. Discuss the inherent ground-truth ambiguity at boundaries — annotators disagree, capping achievable mIoU. What does that mean for interpreting a "good" score?

**Deliverable:** a failure-mode gallery, the cross-entropy-vs-Dice comparison with explanation, and a paragraph on boundary-label ambiguity. Understanding *why* and *where* segmentation breaks — and matching the loss to the task — is professional judgment.
