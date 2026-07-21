# Week 4 — Quiz

Ten questions. Answer key below.

**1. The 'cardinal sin' of a data pipeline is:**

- A. Using too many augmentations
- B. Resizing images
- C. Using a GPU
- D. Leakage — information from test/validation reaching training

**2. Normalization statistics should be computed on:**

- A. The whole dataset
- B. The training set only
- C. The test set
- D. Each batch independently forever

**3. When images cluster (many photos of one individual), you should split:**

- A. By group, so all of an individual's images land in one split
- B. Randomly per image
- C. Alphabetically
- D. By file size

**4. On a 90%/10% imbalanced set, overall accuracy:**

- A. Can look high while the rare class's recall is poor
- B. Equals recall
- C. Cannot be computed
- D. Is always reliable

**5. Recall answers the question:**

- A. How many parameters?
- B. Of images predicted X, how many were X?
- C. Of true class-X images, how many did we catch?
- D. How fast is inference?

**6. A medical disease screen should usually prioritize:**

- A. Recall (don't miss true cases)
- B. Top-5 accuracy
- C. Inference speed
- D. Precision

**7. A large train-vs-validation gap indicates:**

- A. A learning rate that is too low
- B. Overfitting
- C. Underfitting
- D. A data leak fixing itself

**8. Early stopping works by:**

- A. Training for a fixed huge number of epochs
- B. Lowering the batch size
- C. Removing augmentation
- D. Keeping the checkpoint from before validation loss started rising

**9. A learning-rate 'warmup' means:**

- A. Training on easy images first
- B. Starting with a tiny LR and ramping up over the first epochs
- C. Pre-heating the GPU
- D. Using a constant high LR

**10. The fastest way to confirm a training bug (vs. a tuning issue) is to:**

- A. Train longer
- B. Add more data
- C. Lower the learning rate
- D. Overfit a tiny subset — a good model should hit ~100% on 10 images

---

## Answer key

1. **D. Leakage — information from test/validation reaching training** — Leakage (e.g. duplicate images across splits) produces fake results that collapse in production.
2. **B. The training set only** — Fitting stats on test data leaks information; use train-only (or standard pretrained) stats.
3. **A. By group, so all of an individual's images land in one split** — Random per-image splits leak near-duplicates across train and test.
4. **A. Can look high while the rare class's recall is poor** — The accuracy trap returns — report per-class metrics to expose it.
5. **C. Of true class-X images, how many did we catch?** — Recall = true positives / actual positives; precision is the other one.
6. **A. Recall (don't miss true cases)** — Missing a real case is costly, so recall is weighted; the metric encodes values.
7. **B. Overfitting** — Great training but poor validation is the signature of overfitting.
8. **D. Keeping the checkpoint from before validation loss started rising** — It halts memorization by reverting to the best-generalizing checkpoint.
9. **B. Starting with a tiny LR and ramping up over the first epochs** — Warmup stabilizes early training, especially for large batches and Transformers.
10. **D. Overfit a tiny subset — a good model should hit ~100% on 10 images** — If it can't memorize a handful, you have a bug (labels, loss, shapes), not a hyperparameter.
