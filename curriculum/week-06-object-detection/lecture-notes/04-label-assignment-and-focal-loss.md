# Lecture 4 — Label assignment and focal loss: the theory that makes one-stage detection work

Two questions sit at the heart of every detector and get far less airtime than architecture diagrams:
*which predictions are responsible for which objects* (label assignment), and *how do you train when 99.9% of
candidates are background* (the imbalance problem). This lecture makes both precise and derives **focal loss**
(Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017) — the single idea that let a one-stage
detector match two-stage accuracy, and a template for imbalance-aware losses generally.

## The label-assignment problem, stated

A dense detector emits, say, 100k candidate predictions (locations × anchors × scales). Ground truth has maybe
5 boxes. Training needs a target for *every* candidate: which are positives (regress to which GT, classify as
which class) and which are negatives (classify as background). This mapping is **label assignment**, and it is
a genuine design axis, not a detail:

- **Fixed IoU assignment (RetinaNet, Faster R-CNN):** anchor is a positive for the GT of maximum IoU if that
  IoU ≥ 0.5, negative if its best IoU < 0.4, ignored in between. Simple, but the thresholds are hyperparameters
  and small objects may match no anchor at all.
- **Center-based (FCOS):** a location is positive if it falls inside a GT box (and, with center-sampling, near
  its center); it regresses the four side distances. Removes anchors entirely.
- **Optimal / dynamic assignment (OTA, Ge et al. 2021; SimOTA in YOLOX):** treat assignment as an optimal-
  transport problem — each GT "supplies" a dynamic number of positive predictions chosen to minimize a combined
  classification+localization cost. State-of-the-art detectors moved here because fixed thresholds are
  suboptimal per object.

The through-line: assignment decides *what the loss even means*, and better assignment often beats a fancier
backbone. This is why the same architecture reports different AP under different assignment rules.

## The imbalance problem, quantified

In RetinaNet's dense grid, each image yields ~100k anchors of which perhaps 10-100 are foreground. That is a
**~1000:1 background:foreground ratio**. With ordinary cross-entropy, even a *tiny* per-example loss on each
easy background box, summed over ~100k of them, dwarfs the foreground loss — the gradient is dominated by easy
negatives the model already classifies correctly, and training stalls or collapses to "predict background
everywhere." Two-stage detectors dodge this because the RPN pre-filters to ~1-2k proposals with a manageable
ratio. One-stage detectors face it head-on.

## Deriving focal loss

Write binary cross-entropy for the correct-class probability `p_t` (`p_t = p` if the object is present, `1−p`
if not):

    CE(p_t) = -log(p_t).

Its problem: a well-classified example with `p_t = 0.9` still contributes `−log(0.9) ≈ 0.105`, and summed over
100k easy negatives that is a large, useless gradient. **Focal loss** adds a **modulating factor** `(1 − p_t)^γ`
with a tunable focusing parameter `γ ≥ 0`:

    FL(p_t) = -(1 - p_t)^γ · log(p_t).

Read the factor. For a **well-classified** example (`p_t → 1`), `(1 − p_t)^γ → 0`, so its loss is driven
toward zero — easy examples are silenced. For a **misclassified** example (`p_t` small), `(1 − p_t)^γ → 1`, so
the loss is essentially unchanged — hard examples keep their full gradient. Concretely at `γ = 2`: an easy
negative at `p_t = 0.9` is down-weighted by `(0.1)² = 0.01` — a **100× reduction** — while a hard example at
`p_t = 0.1` is barely touched (`0.9² = 0.81`). The relative contribution of hard examples rises by orders of
magnitude without any hard-mining heuristic. In practice an `α`-balanced form `FL = −α_t (1 − p_t)^γ log p_t`
adds a class-frequency weight; the paper uses `γ = 2, α = 0.25`.

## Why this is more than a trick

Focal loss reframes imbalance as a *loss-shaping* problem rather than a *sampling* problem. Hard-negative
mining (SSD) and balanced sampling (Faster R-CNN) fix the ratio by *discarding data*; focal loss keeps every
example but reweights by difficulty, which is smoother and needs no ratio hyperparameter. The same idea recurs
whenever a task is imbalanced — it is a general tool, not a detection curiosity. The paper's headline result:
RetinaNet, a *one-stage* detector, trained with focal loss, **surpassed** the two-stage Faster R-CNN/FPN of its
day in both speed and accuracy — direct evidence that the imbalance, not the single-stage architecture, had
been the accuracy gap all along.

## A worked contribution comparison

Suppose an image has 100 hard foreground examples at `p_t = 0.3` and 100,000 easy background examples at
`p_t = 0.95`. Under plain CE: foreground total ≈ `100 × 1.20 = 120`; background total ≈ `100000 × 0.051 =
5130` — background is ~43× the foreground signal. Under FL (`γ = 2`): each background term is scaled by
`0.05² = 0.0025`, so background total ≈ `5130 × 0.0025 ≈ 12.8`, while foreground scales by `0.7² = 0.49` to
≈ `59` — now **foreground dominates** the gradient. That inversion, achieved with one exponent, is the whole
result.

## Pitfalls

- **Setting `γ` too high** over-suppresses even moderately hard negatives and starves the classifier; `γ = 2`
  is the robust default, tuned per dataset.
- **Applying focal loss where the ratio is already balanced** (e.g. the second stage of a two-stage detector)
  helps little and can hurt — it is a fix for *dense* prediction.
- **Ignoring assignment when comparing losses.** A focal-loss detector with a poor assignment rule can lose to
  a cross-entropy detector with a great one; the two axes interact.

**Takeaway:** label assignment (which candidate is responsible for which object) and class imbalance (background
outnumbers foreground ~1000:1 in dense detectors) are the two problems that actually decide one-stage accuracy.
Focal loss `FL(p_t) = −(1−p_t)^γ log p_t` reshapes the loss so easy negatives are silenced (factor → 0) and
hard examples keep full weight, inverting the gradient balance — no sampling needed. Modern detectors push
assignment further into dynamic/optimal-transport rules. Get assignment and imbalance right and the
architecture matters less than you think.
