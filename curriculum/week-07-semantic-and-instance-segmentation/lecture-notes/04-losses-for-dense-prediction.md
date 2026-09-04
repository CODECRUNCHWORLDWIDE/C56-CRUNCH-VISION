# Lecture 4 — The loss zoo for dense prediction: from weighted cross-entropy to Lovász-Softmax

The metric you report (Lecture 3) is not the objective you can train on — mIoU and PQ are non-differentiable
(they involve counting and matching). So dense-prediction training is a study in **surrogate losses**: differentiable
objectives that, when minimized, drive the true metric up, while surviving the brutal class imbalance of pixel data.
This lecture derives the main losses, their gradients, and when each is right. Getting the loss wrong is the single most
common reason a segmenter "trains fine" yet misses the small thing you actually cared about.

## The baseline: per-pixel cross-entropy, and why it fails on imbalance

Treating pixels as independent, the default loss is the mean per-pixel cross-entropy:

    L_CE = −(1/|Ω|) Σ_{p∈Ω} log softmax(z_p)[y_p].

Its gradient at pixel `p` for class `k` is the familiar `softmax(z_p)[k] − 1[k=y_p]`. The failure is structural: the sum
is over pixels, so if 99% of pixels are background, 99% of the gradient signal is about background. The network
minimizes L_CE efficiently by getting background right and the tiny foreground wrong — precisely the accuracy trap in
loss form. Two families fix this: **re-weight the pixels**, or **optimize an overlap surrogate**.

## Fix 1 — re-weight: class-weighted CE and focal loss

**Weighted cross-entropy** multiplies each pixel's loss by a class weight `w_{y_p}`, e.g. inverse-frequency or the
median-frequency balancing of Eigen & Fergus (ICCV 2015). This directly upweights rare classes; the risk is
instability and boundary over-prediction if weights are extreme.

**Focal loss** (Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017), born for detection's foreground–
background imbalance and heavily used in segmentation, reshapes CE to down-weight *easy* pixels:

    L_focal = −(1 − p_t)^γ · log p_t,     p_t = softmax(z_p)[y_p].

For a well-classified pixel `p_t → 1`, the modulating factor `(1−p_t)^γ → 0`, so easy background contributes almost
nothing and the gradient concentrates on hard, ambiguous pixels (boundaries, rare classes). `γ≈2` is standard. Focal is
"soft hard-example mining" with no sampling heuristics.

## Fix 2 — optimize overlap: soft-Dice, and why it directly targets the metric

Since Dice *is* the metric in medical imaging, optimize a differentiable relaxation. Replace the hard mask with the
softmax probability `p_p ∈ [0,1]` and use the **soft-Dice loss** (Milletari et al., *V-Net*, 3DV 2016):

    L_Dice = 1 − (2 Σ_p p_p g_p + ε) / (Σ_p p_p + Σ_p g_p + ε),

with `g_p ∈ {0,1}` the ground truth and `ε` a smoothing term that both prevents division by zero and defines the loss
when a class is absent. Because the class fraction cancels in the ratio, Dice loss is **scale-invariant to class size** —
a 100-pixel tumor and a 100000-pixel organ contribute comparably. That is exactly why it survives imbalance where CE
drowns. Pitfalls: soft-Dice has a noisier, less well-behaved gradient than CE (the denominator couples all pixels),
can be unstable early in training, and is undefined-ish for empty classes — which is why practitioners almost always use
**CE + Dice** (or focal + Dice) combined, getting CE's stable gradients and Dice's imbalance robustness. nnU-Net's
default is exactly Dice + CE.

## Fix 3 — optimize IoU directly: Lovász-Softmax

Can we optimize IoU (Jaccard) itself, not just Dice? The Jaccard index over discrete sets is not differentiable, but
Berman, Triki & Blaschko (*The Lovász-Softmax loss*, CVPR 2018) show the Jaccard *loss* — as a function of the vector of
per-pixel errors — is a **submodular set function**, and every submodular function has a tight, piecewise-linear convex
surrogate: its **Lovász extension**. Sorting the pixel errors and applying the Lovász extension yields a differentiable
surrogate whose minimization directly targets IoU, typically beating CE on mIoU for the classes it is applied to. It is
more expensive (a per-image sort) and fiddlier than Dice, but it is the principled "optimize the metric" loss.

## Fix 4 — attack the boundary directly

Because error concentrates at boundaries (Lecture 3), **boundary losses** add a term that specifically penalizes edge
mismatch. The **Boundary loss** of Kervadec et al. (*MIDL 2019, Medical Image Analysis 2021*) uses a distance-transform
weighting so a misclassified pixel far from the true boundary is penalized more than a near one, giving a smooth,
integral-based surrogate for boundary distance that pairs well with a regional loss (Dice). This is standard in medical
segmentation where a 2-pixel margin error can be clinically decisive.

## Choosing the loss — a decision procedure

1. **Balanced classes, plenty of data** → plain (optionally class-weighted) cross-entropy is fine.
2. **Imbalanced foreground/background, hard examples** → focal, or CE + focal.
3. **Small target, medical, you report Dice** → **CE + soft-Dice** (the robust default); add a boundary loss if margins
   matter clinically.
4. **You are judged on mIoU and want to squeeze it** → add **Lovász-Softmax**.
5. Always sanity-check the loss against the metric: if Dice-on-validation stalls while CE keeps dropping, the network is
   optimizing the wrong thing (getting confident on background) — a diagnostic you will use in Challenge 2.

**Takeaway:** you cannot train on mIoU or PQ (non-differentiable), so dense prediction uses surrogate losses that also
survive class imbalance. Cross-entropy is dominated by background; fix it by re-weighting (class-weighted CE, focal —
which down-weights easy pixels by `(1−p_t)^γ`) or by optimizing an overlap surrogate (soft-Dice — scale-invariant to
class size, hence imbalance-robust; Lovász-Softmax — the differentiable Lovász extension of the Jaccard set function,
directly targeting IoU). In practice combine a regional loss (Dice) with a stable pixel loss (CE) and, when boundaries
are decisive, a distance-based boundary term — matching the loss to the imbalance and to the metric you must report.
