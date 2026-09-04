# Exercise 2 — Train a frozen feature extractor (linear probe)

**Goal:** transfer learning the fast, small-data-safe way, treated as the serious method it is.

## Tasks

1. Freeze all backbone parameters (`requires_grad = False`); leave only the new head trainable. Verify by
   counting trainable vs. total parameters.
2. Train just the head on your small dataset with augmentation. Using the cached features from Exercise 1,
   train a linear probe directly on `phi(x)` and confirm it is dramatically faster than forwarding through the
   backbone every epoch.
3. Report validation accuracy and the train/validation gap. Note how little it overfits, and explain why in
   one sentence (the backbone cannot memorize your images because it does not update).
4. Time the training and compare to what training a CNN from scratch (Week 3) took for similar accuracy.
5. **Probe depth.** Extract features from an *earlier* backbone stage (not just the final pooled vector) and
   linear-probe those too. Report whether earlier or later features win on your data, and connect the result to
   Lecture 1's general-to-specific hierarchy.

## Deliverable

A notebook training the frozen-backbone model, with accuracy, the train/val gap, and training time; the
cached-feature speedup; and the depth-probe comparison with a two-sentence interpretation tying it to the
feature hierarchy.
