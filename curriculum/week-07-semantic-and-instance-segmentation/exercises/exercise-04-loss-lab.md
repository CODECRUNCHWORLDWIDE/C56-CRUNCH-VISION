# Exercise 4 — Loss lab: cross-entropy vs. Dice vs. focal on an imbalanced target

**Goal:** internalize Lecture 4 — that the loss must match the imbalance and the metric — by implementing the
losses and watching them behave differently on a deliberately imbalanced task.

## Part A — implement (paper + code)

1. Implement, in PyTorch, `soft_dice_loss(logits, target)` and `focal_loss(logits, target, gamma=2)` for binary
   segmentation. For soft-Dice, use the probability (post-sigmoid/softmax), include the `ε` smoothing term, and explain in
   a comment what `ε` protects against.
2. Verify your focal loss reduces to standard binary cross-entropy when `gamma = 0`, with `allclose` against
   `F.binary_cross_entropy_with_logits`.
3. Derive on paper the gradient of soft-Dice w.r.t. a single pixel probability and note why it couples all pixels (the
   denominator depends on every pixel), unlike CE whose per-pixel gradient is local.

## Part B — the imbalance experiment

1. Build (or take) a small binary-segmentation dataset where the foreground is a **small fraction** of pixels (<5%) —
   synthetic small shapes on large backgrounds are fine and fully reproducible.
2. Train the *same* tiny U-Net three times, identical except the loss: (i) plain cross-entropy, (ii) soft-Dice, (iii)
   CE + Dice combined. Fix seeds.
3. Report **foreground IoU/Dice** (not pixel accuracy) on a held-out split for each, and overlay the predicted masks.
   Show that plain CE tends to under-segment or miss the small foreground while Dice/CE+Dice recover it — and explain
   *why* in terms of which pixels dominate the CE gradient.

## Deliverable

A notebook with tested `soft_dice_loss`/`focal_loss` (including the `gamma=0`→BCE check), the paper gradient note, and the
three-way loss comparison with foreground IoU/Dice and mask overlays — plus one paragraph connecting the observed
difference to the class-imbalance argument of Lecture 4.
