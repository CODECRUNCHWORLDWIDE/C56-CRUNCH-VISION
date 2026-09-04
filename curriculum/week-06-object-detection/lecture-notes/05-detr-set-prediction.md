# Lecture 5 — Set prediction: DETR, Hungarian matching, and detection without anchors or NMS

Every detector so far shares two hand-designed components: **anchors** (or their center-based
surrogates) and **non-maximum suppression**. Both are heuristics bolted onto the network to convert dense,
duplicated predictions into a clean set. **DETR** — the DEtection TRansformer (Carion et al., "End-to-End
Object Detection with Transformers," ECCV 2020) — deletes both by reframing detection as **direct set
prediction** trained with a bipartite-matching loss. This is the paradigm the field moved toward, and its core
idea — a set loss that is invariant to prediction order — is worth understanding precisely.

## Detection as set prediction

DETR predicts a **fixed-size set** of `N` predictions (N larger than the max objects per image, e.g. 100),
each a (class, box) pair, where "class" includes a special **∅ (no object)** label. A CNN backbone extracts
features; a Transformer encoder-decoder with `N` learned **object queries** attends over the image and emits
the `N` predictions **in parallel** — no anchors, no proposals, no per-location grid. The output is genuinely a
*set*: there is no meaningful order to the `N` predictions, and the loss must respect that.

## The bipartite-matching loss

The central difficulty: how do you compute a loss between an *unordered* predicted set and the ground-truth
set, when prediction 7 might correspond to GT object 2? You cannot use a fixed positional target. DETR's answer
is to first find the **optimal one-to-one assignment** between predictions and ground truths, then supervise
each prediction by its matched GT (padding GT to size `N` with ∅).

Formally, let `σ` be a permutation of the `N` predictions. Find the permutation minimizing total matching cost:

    σ* = argmin_σ  Σ_i  L_match( y_i , ŷ_{σ(i)} )

where each ground truth `y_i` (or ∅) is matched to one prediction, and `L_match` combines a class-probability
term and a box term (a mix of L1 and **GIoU** on the boxes). This is exactly the **assignment problem**, solved
in `O(N³)` by the **Hungarian algorithm** (Kuhn, "The Hungarian Method," 1955) — a classic combinatorial
optimization, here run once per image to decide targets. Given `σ*`, the training loss is the sum over matched
pairs of a classification (cross-entropy, including ∅) plus box (L1 + GIoU) loss.

## Why this removes NMS and anchors

The bipartite matching is **one-to-one**: each ground truth is assigned to **exactly one** prediction, and
every other prediction is pushed toward ∅. This *directly penalizes duplicates during training* — if two
predictions fire on the same object, only one can be matched; the other is trained to say "no object." So the
network **learns not to emit duplicates**, and NMS becomes unnecessary at inference. Likewise, because object
queries learn their own spatial specializations from data, there are **no anchors** to tune. DETR is thus the
first truly **end-to-end** detector: image in, set of boxes out, with no post-processing heuristic in the loop.

```mermaid
flowchart LR
  A["Image"] --> B["CNN backbone"]
  B --> C["Transformer encoder-decoder + N object queries"]
  C --> D["N predictions, in parallel"]
  D --> E["Hungarian bipartite match to ground truth"]
  E --> F["Per-pair class + L1 + GIoU loss"]
```
*DETR: parallel set prediction supervised through one-to-one Hungarian matching — no anchors, no NMS.*

## Honest limitations, and what came next

The original DETR paid for its elegance:

- **Slow convergence.** It needed ~500 training epochs on COCO (vs. ~12-36 for Faster R-CNN) because the
  matching is unstable early and cross-attention over dense feature maps is hard to learn. **Deformable DETR**
  (Zhu et al., ICLR 2021) fixed this with sparse deformable attention, cutting training ~10× and improving small
  objects.
- **Weak small-object AP** initially, since the single low-resolution feature map hurt tiny instances —
  addressed by multi-scale deformable attention.
- **Set-loss instability** — later work (DN-DETR, DINO) added denoising and better query designs to stabilize
  matching.

The trajectory matters: within two years, DETR-family detectors (DINO; Zhang et al., 2022) reached
state-of-the-art COCO AP, and the *concept* — one-to-one assignment removing NMS — propagated back into CNN
detectors (e.g. one-to-one-trained YOLOs). Set prediction is not merely an alternative; it reframed what a
detector's output and loss *are*.

## Worked intuition: why one-to-one changes the gradient

Under the usual dense loss, many anchors around an object are all positives — the network is *rewarded* for
firing redundantly, and NMS mops up afterward. Under DETR's one-to-one loss, exactly one prediction is rewarded
per object and all near-duplicates are pushed to ∅ — the network receives a gradient that actively *discourages*
redundancy. That single change of the assignment cardinality (one-to-many → one-to-one) is what makes the
post-processing NMS step disappear. When you later meet "NMS-free" CNN detectors, they are importing exactly
this idea.

**Takeaway:** DETR reframes detection as direct **set prediction** — a fixed-size set of (class, box) with a
∅ label — trained by finding the optimal one-to-one **Hungarian** match between predictions and ground truth,
then applying a per-pair class + L1 + GIoU loss. Because the match is one-to-one, duplicates are penalized in
training, so **anchors and NMS both vanish**. The price was slow convergence and weak small objects, fixed by
Deformable/DINO successors that now lead COCO. The lasting idea: change assignment from one-to-many to
one-to-one and the hand-designed post-processing disappears.
