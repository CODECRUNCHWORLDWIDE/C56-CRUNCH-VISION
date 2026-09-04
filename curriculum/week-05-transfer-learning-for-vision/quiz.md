# Week 5 — Quiz

Fifteen questions spanning the representation basis of transfer, the feature-extraction/fine-tuning decision, disciplined fine-tuning, the H-divergence adaptation bound, and modern self-supervised/vision-language backbones. Attempt each before the answer key.

**1. Transfer learning works primarily because a deep vision model's early/middle features are:**

- A. general properties of natural images (edges, colour, textures, parts), reusable across tasks
- B. identical to the final classifier head
- C. useful only for the exact pretraining classes
- D. random until fine-tuned on the target

<details>
<summary>Answer</summary>

**A. general properties of natural images (edges, colour, textures, parts), reusable across tasks** — Yosinski (2014) and Zeiler & Fergus (2014) show early filters are near-universal Gabor/colour detectors; only late layers specialize.

</details>

**2. Yosinski et al. (2014) identified two reasons deep-layer features transfer worse; they are:**

- A. specificity to the source classes and fragile co-adaptation between layers
- B. vanishing and exploding gradients
- C. overfitting and underfitting
- D. batch-norm and dropout interference

<details>
<summary>Answer</summary>

**A. specificity to the source classes and fragile co-adaptation between layers** — Deep layers become source-specific AND co-adapted with neighbours, so splitting the stack mid-way hurts; fine-tuning heals both.

</details>

**3. Feature extraction (linear probing) means:**

- A. freezing the backbone and training only a classifier on its fixed features
- B. using no pretrained weights at all
- C. deleting the backbone and keeping the head
- D. training the whole network from scratch

<details>
<summary>Answer</summary>

**A. freezing the backbone and training only a classifier on its fixed features** — The frozen backbone is a fixed feature map phi(x); only the head learns, so it cannot overfit the backbone and features can be cached.

</details>

**4. Kumar et al. (2022) showed that, when pretrained features are already good, full fine-tuning can:**

- A. never overfit regardless of data size
- B. always beat linear probing on every metric
- C. distort the features and underperform a linear probe out-of-distribution
- D. eliminate the need for preprocessing

<details>
<summary>Answer</summary>

**C. distort the features and underperform a linear probe out-of-distribution** — Fine-tuning from a random head can distort robust features; hence LP-FT (probe first, then fine-tune) preserves OOD accuracy.

</details>

**5. The recommended order in the responsible recipe is:**

- A. unfreeze everything immediately with a from-scratch learning rate
- B. never train the head at all
- C. linear-probe the head first, then unfreeze top blocks with a small warmed-up LR
- D. train only the first convolutional layer

<details>
<summary>Answer</summary>

**C. linear-probe the head first, then unfreeze top blocks with a small warmed-up LR** — A random head emits huge gradients; probing first avoids shocking the backbone, then top-down fine-tuning adapts task-specific layers.

</details>

**6. Catastrophic forgetting during fine-tuning happens when:**

- A. you use data augmentation
- B. you freeze the backbone
- C. the dataset is too large
- D. a too-large learning rate overwrites the useful pretrained features with noisy early gradients

<details>
<summary>Answer</summary>

**D. a too-large learning rate overwrites the useful pretrained features with noisy early gradients** — Big early gradients from a random head plus a high LR destroy pretrained knowledge, leaving you worse than a frozen extractor.

</details>

**7. A pretrained backbone requires inputs preprocessed:**

- A. with ImageNet mean/std regardless of the backbone
- B. always converted to grayscale
- C. exactly as its own training data (matching size and per-channel normalization statistics)
- D. at any size and normalization

<details>
<summary>Answer</summary>

**C. exactly as its own training data (matching size and per-channel normalization statistics)** — Mismatched normalization silently corrupts every feature; CLIP/SSL backbones use their own stats, so read weights.transforms().

</details>

**8. The Ben-David et al. (2010) bound writes target error as source error plus:**

- A. the learning rate times the batch size
- B. the entropy of the source labels
- C. the number of target classes
- D. half the H-DeltaH feature divergence plus the best-joint-hypothesis term lambda

<details>
<summary>Answer</summary>

**D. half the H-DeltaH feature divergence plus the best-joint-hypothesis term lambda** — eps_T <= eps_S + (1/2) d_{HdH}(D_S,D_T) + lambda; the three terms are source error, feature divergence, and adaptability.

</details>

**9. In the adaptation bound, the divergence term d_{H Delta H} can be estimated from unlabeled data by:**

- A. training a classifier to distinguish source from target features; its accuracy above chance is the proxy
- B. measuring the pretraining top-1 accuracy
- C. counting the number of target classes
- D. computing the L2 norm of the weights

<details>
<summary>Answer</summary>

**A. training a classifier to distinguish source from target features; its accuracy above chance is the proxy** — A separable domain classifier means large divergence; domain-adversarial training (DANN) drives this estimate toward zero.

</details>

**10. The lambda term (best joint hypothesis) in the bound is large precisely under:**

- A. concept shift, where p(y|x) changes so no single head is good on both domains
- B. the use of cosine learning-rate schedules
- C. any change in batch size
- D. covariate shift, where only p(x) changes

<details>
<summary>Answer</summary>

**A. concept shift, where p(y|x) changes so no single head is good on both domains** — Concept shift changes the label relationship, so lambda is large and feature alignment cannot help; covariate shift keeps lambda small.

</details>

**11. Discriminative / layer-wise-decayed learning rates mean:**

- A. a smaller LR for early (general) layers and larger for late (specific) layers and the head
- B. random per-parameter learning rates
- C. a single LR shared by all layers
- D. no learning rate at all

<details>
<summary>Answer</summary>

**A. a smaller LR for early (general) layers and larger for late (specific) layers and the head** — Early universal features should barely move; task-specific layers adapt more, so LR increases from the input toward the head.

</details>

**12. CLIP (Radford et al., 2021) enables zero-shot classification because it:**

- A. aligns image and text embeddings, so class names can be embedded as text and matched to images
- B. uses no pretraining data
- C. reconstructs masked image patches
- D. was trained on labeled ImageNet only

<details>
<summary>Answer</summary>

**A. aligns image and text embeddings, so class names can be embedded as text and matched to images** — CLIP's joint image-text contrastive training lets you classify into new categories by nearest text-prompt embedding, no target labels.

</details>

**13. Compared with MAE, DINO/DINOv2 features tend to:**

- A. require full fine-tuning to be usable at all
- B. ignore preprocessing entirely
- C. only work on medical images
- D. be strong for frozen linear probing, favouring feature extraction

<details>
<summary>Answer</summary>

**D. be strong for frozen linear probing, favouring feature extraction** — Contrastive/self-distillation SSL yields strong frozen features (great linear probes); masked-generative MAE is weaker frozen but strong fine-tuned.

</details>

**14. WiSE-FT (Wortsman et al., 2022) improves robustness by:**

- A. increasing the learning rate late in training
- B. training a second backbone from scratch
- C. interpolating the weights of the zero-shot and fine-tuned models
- D. removing all augmentation

<details>
<summary>Answer</summary>

**C. interpolating the weights of the zero-shot and fine-tuned models** — theta = (1-alpha)theta_zeroshot + alpha theta_finetuned recovers much ID gain while keeping OOD robustness — near-free weight averaging.

</details>

**15. Under pretraining, an inflated test accuracy most often signals:**

- A. too much weight decay
- B. using a validation set at all
- C. an input size that is too small
- D. test images overlapping the pretraining set (memorization), inflating the number

<details>
<summary>Answer</summary>

**D. test images overlapping the pretraining set (memorization), inflating the number** — Web-scale pretraining makes test leakage the default; audit for near-duplicates or at minimum state the overlap risk explicitly.

</details>

---
