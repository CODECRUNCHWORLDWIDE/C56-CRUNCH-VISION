# Week 4 — Quiz

Fifteen questions, graduate level. Answer key below; each answer explains the reasoning.

**1. The 'cardinal sin' of a data pipeline is:**

- A. Leakage — information from validation/test reaching training
- B. Resizing images to a fixed size
- C. Training on a GPU
- D. Using too many augmentations

<details>
<summary>Answer</summary>

**A. Leakage — information from validation/test reaching training** — Leakage breaks the independence of the test estimate from training, biasing the reported risk optimistically; it survives any amount of data.

</details>

**2. Test accuracy `R̂_T` should be understood as:**

- A. the exact population error rate of the model
- B. a point estimate of the population risk, with its own standard error
- C. a property of the architecture independent of the sample
- D. a value that cannot be given a confidence interval

<details>
<summary>Answer</summary>

**B. a point estimate of the population risk, with its own standard error** — R̂_T is a mean of m Bernoulli trials — an estimator of R(θ) with SE ~sqrt(p(1-p)/m); it must carry error bars.

</details>

**3. When a dataset has 20 photos of the same individual, the correct split is:**

- A. grouped, so all of an individual's images land in one split
- B. random per image so each split is i.i.d.
- C. by file size
- D. alphabetical by filename

<details>
<summary>Answer</summary>

**A. grouped, so all of an individual's images land in one split** — A per-image split scatters near-duplicates across train and test; the model memorizes the individual and the test estimate is inflated. Split by group.

</details>

**4. Normalization statistics (channel mean/std) should be computed on:**

- A. the training split only (or standard pretrained stats)
- B. the test split
- C. each batch forever, never fixed
- D. the whole dataset including test

<details>
<summary>Answer</summary>

**A. the training split only (or standard pretrained stats)** — Fitting stats on data that includes test leaks target-adjacent information; fit on train, then apply the fixed transform to val/test.

</details>

**5. A deep network reports 99% confidence at 80% accuracy. This is a failure of:**

- A. the confusion matrix
- B. calibration (the model is over-confident)
- C. recall
- D. the learning-rate schedule

<details>
<summary>Answer</summary>

**B. calibration (the model is over-confident)** — Calibration means confidence matches accuracy; modern nets are systematically over-confident (Guo et al., 2017), measured by reliability diagrams and ECE.

</details>

**6. Temperature scaling fixes over-confidence by:**

- A. learning a scalar T on validation and using logits z/T, which changes no argmax
- B. increasing the batch size
- C. removing the softmax
- D. retraining the whole network with more data

<details>
<summary>Answer</summary>

**A. learning a scalar T on validation and using logits z/T, which changes no argmax** — Dividing logits by a learned T>1 softens probabilities without changing the predicted class, so accuracy is unchanged while ECE drops.

</details>

**7. On a 90%/10% imbalanced set, overall accuracy:**

- A. cannot be computed
- B. can look high while the rare class's recall is poor
- C. equals the rare class's recall
- D. is always the fairest metric

<details>
<summary>Answer</summary>

**B. can look high while the rare class's recall is poor** — A constant majority prediction scores 90%; only per-class (disaggregated) metrics expose the abandoned tail.

</details>

**8. Focal loss down-weights easy examples via the factor:**

- A. (1 − p_t)^γ, which vanishes as the true-class probability p_t → 1
- B. the learning rate times the batch size
- C. the softmax temperature
- D. a constant equal to the class frequency

<details>
<summary>Answer</summary>

**A. (1 − p_t)^γ, which vanishes as the true-class probability p_t → 1** — FL = −(1−p_t)^γ log p_t; confident-correct examples (p_t→1) contribute almost nothing, focusing training on hard, often rare, cases (Lin et al., 2017).

</details>

**9. SGD's mini-batch gradient relative to the full-batch gradient is:**

- A. a biased estimate that must be corrected
- B. an unbiased estimate with variance scaling like 1/|B|
- C. always exactly equal
- D. only valid for convex losses

<details>
<summary>Answer</summary>

**B. an unbiased estimate with variance scaling like 1/|B|** — E_B[g_B] = ∇R̂, so the mini-batch gradient is unbiased; its covariance ~1/|B| makes the noise scale η/sqrt(|B|).

</details>

**10. The local divergence cliff for the learning rate on a quadratic with top curvature λ_max is at:**

- A. η = λ_max
- B. η = 1/n (n = dataset size)
- C. η equal to the batch size
- D. η ≥ 2/λ_max, where steps amplify along the stiff direction

<details>
<summary>Answer</summary>

**D. η ≥ 2/λ_max, where steps amplify along the stiff direction** — GD contracts error by |1 − η λ_i|; the stiff direction diverges once η ≥ 2/λ_max, which is why 'loss went NaN' usually means LR too high.

</details>

**11. AdamW differs from Adam-with-weight-decay because it:**

- A. decouples the weight-decay term so it is not scaled by the adaptive denominator
- B. removes momentum
- C. normalizes the inputs
- D. uses a larger learning rate

<details>
<summary>Answer</summary>

**A. decouples the weight-decay term so it is not scaled by the adaptive denominator** — In plain Adam the L2 penalty flows through the 1/sqrt(v̂) scaling and is distorted; AdamW applies θ ← θ − ηλθ directly (Loshchilov & Hutter, 2019).

</details>

**12. The experimentally supported explanation for why batch normalization helps is that it:**

- A. smooths the loss landscape, reducing gradient Lipschitzness and permitting larger learning rates
- B. eliminates internal covariate shift entirely
- C. replaces the need for a learning-rate schedule
- D. adds parameters that memorize the data

<details>
<summary>Answer</summary>

**A. smooths the loss landscape, reducing gradient Lipschitzness and permitting larger learning rates** — Santurkar et al. (2018) showed the covariate-shift story is not the mechanism; BN makes the loss/gradients smoother, enabling faster, more stable training.

</details>

**13. Momentum improves convergence on an ill-conditioned quadratic by turning the step count from:**

- A. O(κ²) into O(κ)
- B. O(n) into O(log n)
- C. O(sqrt(κ)) into O(κ)
- D. O(κ) into O(sqrt(κ))

<details>
<summary>Answer</summary>

**D. O(κ) into O(sqrt(κ))** — Heavy-ball/Nesterov acceleration gives the quadratic speedup O(sqrt(κ)); momentum averages gradients, damping oscillation and accelerating consistent directions.

</details>

**14. Aggregate accuracy can hide large disparities across demographic subgroups. The correct response is to:**

- A. remove the rare subgroups from the test set
- B. report only the average, which is the fairest single number
- C. disaggregate metrics by subgroup and treat a large disparity as a defect
- D. increase the learning rate

<details>
<summary>Answer</summary>

**C. disaggregate metrics by subgroup and treat a large disparity as a defect** — Buolamwini & Gebru (2018) found up-to-34-point error gaps hidden behind high aggregate accuracy; subgroup-level reporting is required for classifiers touching people.

</details>

**15. The fastest way to confirm a training *bug* (versus a tuning issue) is to:**

- A. add more data
- B. overfit a tiny subset — a correct model should hit ~100% on ~10 images
- C. lower the learning rate and wait
- D. train for many more epochs

<details>
<summary>Answer</summary>

**B. overfit a tiny subset — a correct model should hit ~100% on ~10 images** — If a model cannot memorize a handful of images, the fault is a bug (labels, loss, shapes, detached graph), not a hyperparameter — the single most useful sanity check.

</details>

---
