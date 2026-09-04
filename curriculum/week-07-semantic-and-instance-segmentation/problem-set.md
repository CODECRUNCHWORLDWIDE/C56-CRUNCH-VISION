# Week 7 — Graduate Problem Set: Dense Prediction, Metrics, and Losses

Ten problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution sketches
follow each — attempt the problem fully before reading them. Notation: `P` = predicted pixel set for a class, `G` =
ground-truth set, `TP=|P∩G|`, `FP=|P\G|`, `FN=|G\P|`.

**P1 (warm-up, computation).** A predicted mask has 800 pixels, the ground truth 1000, and they overlap in 600. Compute
IoU, Dice, precision, and recall.
*Sketch.* Union = 800+1000−600 = 1200, so IoU = 600/1200 = 0.5. Dice = 2·600/(800+1000) = 1200/1800 = 2/3. Precision =
600/800 = 0.75, recall = 600/1000 = 0.6. Check Dice = 2·IoU/(1+IoU) = 1/1.5 = 2/3. ✓

**P2 (proof).** Prove Dice = 2·IoU/(1+IoU) for any non-empty masks, using inclusion–exclusion.
*Sketch.* `|P|+|G| = |P∩G|+|P∪G|`. With `i=|P∩G|, u=|P∪G|`: Dice = 2i/(i+u); divide by u → 2(i/u)/(1+i/u) = 2·IoU/(1+IoU). ∎

**P3 (computation).** Show pixel accuracy overstates quality. An image has 10000 pixels; the object is 200 pixels. A
model predicts *all background*. Compute pixel accuracy and the object's IoU.
*Sketch.* It labels 9800/10000 correct → accuracy 0.98. Object IoU = 0/200 = 0 (TP=0). A 0.98-accurate model that finds
nothing — the accuracy trap.

**P4 (receptive field, computation).** A backbone is four stride-2 3×3-conv stages. Compute the cumulative downsampling
factor and, qualitatively, why a boundary cannot be localized from the final map. Then state how (a) a U-Net skip and (b)
a dilation-rate-2 atrous conv each address it.
*Sketch.* Downsampling = 2⁴ = 16×; the final cell is a 16×16 input patch, so edges within it are unresolvable. (a) A skip
carries the pre-downsample high-res features to the decoder to restore 'where'. (b) Atrous conv enlarges receptive field
at stride 1, so it never loses the resolution in the first place.

**P5 (derivation).** For binary segmentation with per-pixel probability `p` and label `g∈{0,1}`, write soft-Dice loss and
derive ∂L/∂p_j for one pixel j. Explain why the gradient couples all pixels.
*Sketch.* `L = 1 − (2Σp g + ε)/(Σp + Σg + ε)`. Let `N = 2Σpg+ε`, `D = Σp+Σg+ε`. Then ∂L/∂p_j = −(2 g_j D − N)/D². Because
D contains Σp over *all* pixels, every pixel's gradient depends on the global sums — unlike CE's purely local per-pixel
gradient. This is why soft-Dice is noisier and often paired with CE.

**P6 (loss reasoning).** You have a lane-marking dataset where lanes are ~0.5% of pixels. Rank plain CE, weighted CE,
focal, and soft-Dice for foreground IoU, and justify.
*Sketch.* Plain CE worst (gradient dominated by 99.5% background). Weighted CE and focal both help by re-weighting/down-
weighting easy background; focal adapts to hardness automatically. Soft-Dice is scale-invariant to class size and
directly overlap-based, typically best on foreground IoU; CE+Dice combines stability and imbalance robustness — the
practical winner.

**P7 (PQ, computation).** In one image, class "car" has 3 ground-truth and 3 predicted segments. Matches (IoU>0.5) have
IoUs 0.8, 0.7; one predicted and one ground-truth segment are unmatched. Compute SQ, RQ, and PQ.
*Sketch.* TP=2, FP=1, FN=1. SQ = (0.8+0.7)/2 = 0.75. RQ = 2/(2+0.5·1+0.5·1) = 2/3 ≈ 0.667. PQ = SQ·RQ = 0.75·0.667 = 0.5.

**P8 (proof/analysis).** Prove the IoU>0.5 matching rule in PQ yields at most one predicted segment matching each ground
truth (uniqueness), hence no greedy ambiguity.
*Sketch.* Two distinct predictions each with IoU>0.5 to the same G would each cover >half of G's pixels, so they'd share
pixels of G — but predicted panoptic segments are disjoint (partition), contradiction. So at most one prediction exceeds
0.5 IoU with a given G. ∎

**P9 (analysis).** mIoU treats a 50-pixel class and the sky equally. Give a concrete scenario where this makes mIoU
volatile, and argue whether that is a bug or a feature.
*Sketch.* For a 50-pixel class, mislabeling 25 pixels swings its IoU from ~1 to ~0.5, moving the mean noticeably even
though few pixels changed. It is a feature (rare classes matter and shouldn't be hidden) but demands reporting *per-class*
IoU and often multiple images/runs to stabilize the estimate.

**P10 (open, high-stakes).** For a tumor-segmentation model, three radiologists' masks of one lesion have pairwise IoUs
~0.8. Discuss: (i) the ceiling this places on any model's achievable IoU; (ii) how to train/evaluate against a
distribution of labels rather than one; (iii) why calibration is a patient-safety property here.
*Sketch.* (i) If experts agree only to ~0.8, a model at ~0.8 is at the noise floor — not "wrong"; a claimed 0.95 likely
means overfitting to one annotator. (ii) Fuse labels (e.g. STAPLE, Warfield et al. 2004) into a probabilistic consensus,
or train with soft/multi-annotator targets and report against each. (iii) An overconfident boundary can under-dose a
tumor edge; a downstream planner trusts the probabilities, so miscalibration is a clinical risk, requiring reliability
diagrams / temperature scaling and human-in-the-loop review.
