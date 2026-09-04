# Exercise 4 — Label assignment and focal loss, quantified

**Goal:** make the two ideas from Lecture 4 concrete — see how assignment changes targets and how
focal loss reshapes the gradient.

## Part A — label assignment

1. Generate a small synthetic scene: 3 ground-truth boxes and a dense grid of ~2000 anchors of a few
   sizes/ratios over the image.
2. Implement **fixed-IoU assignment**: for each anchor compute its max IoU to any GT; mark positive if ≥ 0.5,
   negative if < 0.4, ignore in between. Report how many anchors are positive, negative, ignored — observe the
   ~hundreds-to-one negative:positive ratio directly.
3. Implement a simple **center-based (FCOS-style)** assignment: an anchor location is positive if it falls
   inside a GT box. Compare the positive sets to the IoU rule and note which small GT gets *no* IoU positive
   but *does* get a center positive.

## Part B — focal loss

1. Implement `focal_loss(p, target, gamma, alpha)` for binary classification and verify that at `gamma = 0` it
   reduces exactly to (alpha-weighted) binary cross-entropy.
2. Reproduce Lecture 4's contribution experiment: 100 hard foreground examples at `p_t = 0.3` and 100,000 easy
   background at `p_t = 0.95`. Compute total foreground vs. background loss under CE and under FL (`gamma = 2`),
   and show that FL **inverts** which term dominates the gradient.
3. Plot FL loss vs. `p_t` for `gamma ∈ {0, 0.5, 1, 2, 5}` and describe how increasing `gamma` suppresses
   easy examples.

## Deliverable

A notebook reporting the assignment counts and the CE-vs-FL contribution inversion with actual numbers, plus
the FL-vs-p_t curves. One paragraph connecting the two parts: why dense one-stage detectors need *both* a
sensible assignment rule and an imbalance-aware loss.
