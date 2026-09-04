# Week 4 — Graduate Problem Set: Estimation, Metrics, Optimization, and the Long Tail

Ten problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution
sketches are at the end — attempt each fully before reading them. Notation: a test set of `m` i.i.d. examples,
model error rate estimate `p̂ = R̂_T`, class probabilities `p = softmax(z)`.

**P1 (accuracy as an estimator).** You measure 88% accuracy on `m = 250` test images. Model the number correct
as Binomial(`m`, `p`). Give the standard error of `p̂` and an approximate 95% confidence interval. How large
must `m` be for the 95% CI half-width to be ≤ 1 percentage point at `p ≈ 0.88`?

**P2 (why grouped splitting).** A dataset has `G` individuals, each with `k` near-identical photos, one class per
individual-group irrelevant. Argue informally why a random per-image 80/20 split makes the test accuracy an
over-estimate of population risk, and why `GroupKFold` fixes it. Identify which condition (same-distribution vs.
independence-from-training) the per-image split violates.

**P3 (precision/recall trade).** A binary classifier thresholds a score at `τ`. Sketch how precision and recall
each move as `τ` increases from 0 to 1, and prove that recall is monotonically non-increasing in `τ`. Where on
the precision–recall curve does F1 tend to be maximized relative to the extremes?

**P4 (macro vs. micro).** With classes of sizes `n_1 ≫ n_2`, construct a 2-class confusion matrix where
micro-F1 is high but macro-F1 is low. State in one sentence which you would report for a rare-disease screen and
why.

**P5 (proper scoring / calibration).** Show that the log loss `−log p_y` and the Brier score are minimized in
expectation by the *true* conditional probabilities (i.e. they are proper scoring rules), whereas 0–1 accuracy
is not. Explain why this makes log loss/Brier sensitive to calibration while accuracy is blind to it.

**P6 (temperature scaling).** With logits `z` and temperature `T`, let `p^{(T)} = softmax(z/T)`. Show that as
`T → ∞` the distribution → uniform and as `T → 0⁺` it → the one-hot `argmax`. Argue why scaling by `T` never
changes the predicted class, hence never changes accuracy, yet can change ECE.

**P7 (SGD noise scale).** Per-example gradients have per-coordinate variance `s²`. Derive the variance of the
mini-batch gradient estimate for batch size `|B|`, and the resulting noise std of a single SGD step of learning
rate `η`. Use it to justify the linear scaling rule (double `|B|` ⇒ roughly double `η`) and state where the
rule breaks (the `2/λ_max` ceiling, requiring warmup).

**P8 (momentum on a quadratic).** For `R̂(θ) = (a/2)θ²` (`a>0`), write the momentum update `v_{t+1} = μ v_t + a
θ_t`, `θ_{t+1} = θ_t − η v_{t+1}`. Express it as a 2×2 linear recurrence in `(θ_t, v_t)` and state the stability
condition on `(η, μ)`. Explain qualitatively why momentum accelerates the low-curvature direction.

**P9 (focal loss gradient).** For binary focal loss `FL(p_t) = −(1−p_t)^γ log p_t` with `p_t = σ(z)` for the
positive class, show that as `p_t → 1` the loss and its gradient in `z` both → 0 faster than for cross-entropy,
and interpret what this does to the training signal from easy vs. hard examples. Take `γ = 2` for concreteness.

**P10 (open — reporting under shift).** Your test set has class proportions `q(y)` but deployment has different
proportions `q'(y)`, with the same class-conditional `p(x|y)`. Show how to re-weight your per-example test
losses by `q'(y)/q(y)` (importance weighting) to estimate deployment risk, and state the assumptions this
requires. Then discuss, in a paragraph, why this correction cannot rescue you if `p(x|y)` *also* shifts (e.g. a
new camera), and what you would report instead. (Open-ended; argue carefully.)

---

## Solution sketches

**S1.** SE `= sqrt(p̂(1−p̂)/m) = sqrt(0.88·0.12/250) ≈ 0.0206`; 95% CI `≈ 0.88 ± 1.96·0.0206 = [0.840, 0.920]`.
For half-width ≤ 0.01: `1.96·sqrt(0.88·0.12/m) ≤ 0.01 ⇒ m ≥ (1.96²·0.1056)/0.0001 ≈ 4056`.
**S2.** The per-image split places near-duplicate photos of the same individual in both train and test, so the
test examples are *not independent of training* — the model recognizes the memorized individual, inflating
accuracy. `GroupKFold` keeps each individual wholly in one fold, restoring independence. It violates the
independence-from-training condition, not the same-distribution one.
**S3.** As `τ↑`, fewer positives are predicted: TP and FP both fall, so recall = TP/(TP+FN) is non-increasing
(FN rises as TP falls with fixed positives); precision generally rises but can be non-monotone. F1 peaks at an
interior `τ` balancing the two, away from the extremes where one term collapses.
**S4.** E.g. class 1 (n=990) perfectly classified, class 2 (n=10) all missed: micro pools TP/FP/FN so micro-F1
≈ 0.99; macro-F1 averages ~1.0 and ~0.0 ⇒ ~0.5. Report macro-F1 for the rare-disease screen — it refuses to let
the tail hide.
**S5.** For a proper scoring rule `S(p, y)`, `E_{y~q}[S(p,y)]` is minimized at `p = q`. Log loss:
`E_q[−log p_y] = H(q) + D_KL(q‖p)`, minimized (KL=0) at `p=q`. Brier: `E‖p − onehot‖²` is a shifted variance,
minimized at `p = q`. Accuracy depends only on `argmax`, so any `p` with the right argmax scores identically —
it cannot see calibration.
**S6.** `softmax(z/T)_k ∝ exp(z_k/T)`; `T→∞` ⇒ exponent→0 ⇒ uniform; `T→0⁺` ⇒ the max dominates ⇒ one-hot.
Scaling all logits by `1/T` preserves their order, so `argmax` (and accuracy) are invariant; but the *values*
change, so binned confidence-vs-accuracy (ECE) changes.
**S7.** `Var(g_B) = s²/|B|` per coordinate (mean of `|B|` i.i.d. terms); step noise std `= η·s/sqrt(|B|)`.
Doubling `|B|` cuts noise std by `1/sqrt2`; to hold exploration constant, roughly double `η`. It breaks once
`η` nears `2/λ_max` (deterministic divergence), which is why large-batch training needs warmup.
**S8.** Recurrence `[θ_{t+1}; v_{t+1}] = M [θ_t; v_t]` with `M = [[1−ηa, −η],[a, μ]]` (up to ordering); stable
iff both eigenvalues of `M` have modulus < 1, giving a joint condition on `(η, μ)` (for the standard form,
convergence needs `0 ≤ μ < 1` and `0 < ηa < 2(1+μ)`). Momentum lets the low-curvature direction accumulate
consistent gradients, effectively multiplying its step by ~`1/(1−μ)`.
**S9.** `∂FL/∂z = (1−p_t)^γ [ γ p_t log p_t + (p_t − 1) ] · (∂p_t/∂z)`-type expression; the modulating factor
`(1−p_t)^γ → 0` as `p_t→1`, so both `FL` and its gradient vanish for easy examples much faster than CE's
`−log p_t` (whose gradient `p_t − 1` is small but the factor is absent). Hard examples (`p_t` small) keep near-
full weight — training focuses on them.
**S10.** With shared `p(x|y)`, deployment risk `E_{q'}[ℓ] = Σ_y q'(y) E[ℓ|y] = E_{q}[ (q'(y)/q(y)) ℓ ]`, so
importance-weight each test example by `w(y)=q'(y)/q(y)` (assumes `q(y)>0` wherever `q'(y)>0`, and identical
class-conditionals). If `p(x|y)` also shifts, the weights cannot fix it — the class-conditional appearance
changed — and you should report that the estimate is invalid under covariate shift, ideally collecting a small
labeled sample from the deployment distribution and quoting metrics on it.
