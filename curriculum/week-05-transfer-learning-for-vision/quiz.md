# Week 5 — Quiz

Ten questions. Answer key below.

**1. Transfer learning works because early/middle CNN features are:**

- A. Random
- B. Only useful for ImageNet
- C. Specific to the exact classes
- D. General across tasks (edges, textures, parts)

**2. To adapt a pretrained classifier to a new task, you first:**

- A. Delete the backbone
- B. Retrain ImageNet
- C. Replace the head with one sized to your number of classes
- D. Remove all convolutions

**3. 'Feature extraction' transfer means:**

- A. Training everything from scratch
- B. Using no pretrained weights
- C. Deleting the head
- D. Freezing the backbone and training only the new head

**4. Fine-tuning should use a learning rate that is:**

- A. Exactly zero
- B. Much smaller than from-scratch
- C. Much larger than from-scratch
- D. Irrelevant

**5. Catastrophic forgetting during fine-tuning happens when:**

- A. A too-high LR overwrites the useful pretrained features
- B. You use augmentation
- C. The dataset is too large
- D. You freeze the backbone

**6. For a small dataset in a domain similar to ImageNet, the best default is:**

- A. Feature extraction (freeze backbone)
- B. Skip pretraining
- C. Fine-tune all layers with a huge LR
- D. Train from scratch

**7. A pretrained backbone requires inputs that are:**

- A. Preprocessed exactly as its training data (size and normalization)
- B. Unnormalized
- C. Any size and normalization
- D. Always grayscale

**8. Discriminative learning rates mean:**

- A. Random LRs
- B. No LR at all
- C. Smaller LR for early (general) layers, larger for late (specific) layers and the head
- D. One LR for all layers

**9. Transfer learning tends to help *least* on:**

- A. Everyday photos
- B. Pet breeds
- C. Domains very different from natural images (e.g. medical, satellite)
- D. Vehicle types

**10. A recommended fine-tuning recipe is to:**

- A. Unfreeze everything immediately with a high LR
- B. Only train the first layer
- C. Never train the head
- D. Train the new head first, then unfreeze top blocks with a small LR

---

## Answer key

1. **D. General across tasks (edges, textures, parts)** — Universal low/mid-level features transfer; only the final layers are task-specific.
2. **C. Replace the head with one sized to your number of classes** — Swap the final classifier head for your classes; keep the pretrained backbone.
3. **D. Freezing the backbone and training only the new head** — The frozen backbone is a fixed feature extractor; only the head learns.
4. **B. Much smaller than from-scratch** — A small LR refines good weights; too large causes catastrophic forgetting.
5. **A. A too-high LR overwrites the useful pretrained features** — Big noisy gradients early can destroy the pretrained knowledge.
6. **A. Feature extraction (freeze backbone)** — Little data can't safely fine-tune, and similar-domain features already fit.
7. **A. Preprocessed exactly as its training data (size and normalization)** — Mismatched preprocessing makes the features meaningless — a silent failure.
8. **C. Smaller LR for early (general) layers, larger for late (specific) layers and the head** — Early universal features barely move; task-specific layers adapt more.
9. **C. Domains very different from natural images (e.g. medical, satellite)** — Distant domains share fewer statistics, so features transfer less and need more fine-tuning.
10. **D. Train the new head first, then unfreeze top blocks with a small LR** — Head-first avoids destabilizing gradients; then adapt from the top down carefully.
