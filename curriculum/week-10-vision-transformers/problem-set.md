# Week 10 — Graduate Problem Set: Patches, Attention, Complexity, and Pretraining

Ten problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution
sketches follow — attempt each fully before reading them. Notation: an image is `(C, H, W)`, patch size `P`,
token count `N = (H/P)(W/P)`, model dimension `D`, per-head dimension `d_k`, heads `h`.

**P1 (token counting).** For a `(3, 384, 384)` image with `P = 16`, how many patch tokens `N` are there? How
many total input tokens including a single [CLS]? If you instead use `P = 32`, by what factor does `N`
change, and by what factor does the attention score matrix `N x N` change in size?

**P2 (conv-equals-embedding).** Prove that an `nn.Conv2d(C, D, kernel_size=P, stride=P)` applied to an image
computes, at each output spatial location, the linear map `E x_p + b` where `x_p` is the flattened patch and
`E` is the reshaped kernel. State exactly where the stride = kernel condition is used.

**P3 (sqrt(d) scale).** Let the entries of query `q` and key `k in R^{d_k}` be independent with mean 0 and
variance 1. Compute the mean and variance of the dot product `q · k`. Explain, using the softmax Jacobian
`diag(p) - p p^T`, why unscaled logits of standard deviation `sqrt(d_k)` cause vanishing gradients, and how
dividing by `sqrt(d_k)` fixes it.

**P4 (permutation equivariance).** Let `Pi` be an `N x N` permutation matrix. Prove
`Attention(Pi X) = Pi Attention(X)`. Then explain in one sentence why this makes positional encodings
mathematically necessary.

**P5 (complexity).** Derive the time cost (in FLOPs, up to constants) of one multi-head self-attention layer
as a function of `N`, `D`, and `h`, and separately the cost of the position-wise MLP with hidden dimension
`4D`. For which regime of `N` does attention dominate the MLP? Give the crossover in terms of `D`.

**P6 (windowed attention).** Swin partitions `N` tokens into non-overlapping windows of `M x M` patches.
Derive the total attention cost across all windows and show it is `Theta(N M^2 D)` — linear in `N` for fixed
`M`. Compare to global attention at `N = 3136`, `M = 7`. Why is the shifted-window step necessary?

**P7 (positional interpolation).** A ViT pretrained at 224px has a learned position table of `14 x 14 = 196`
entries. You want to fine-tune at 448px (`28 x 28 = 784` patches). Describe precisely the operation that
adapts the position table, why naive truncation/zero-padding fails, and what invariance a *relative* position
bias (Swin) would give you instead.

**P8 (attention rollout).** Given per-layer averaged attention matrices `A_1, ..., A_L in R^{(N+1)x(N+1)}`,
write the rollout recursion including the residual term, and explain why `0.5 A_l + 0.5 I` (renormalized) is
used rather than `A_l` alone. What does rollout ignore that a full attribution method would not?

**P9 (MAE mask ratio).** Argue quantitatively why an image MAE uses ~75% masking while a text model (BERT)
uses ~15%. Frame it in terms of patch redundancy and the difficulty of the reconstruction pretext task: what
goes wrong at 15% masking of image patches, and why does He et al.'s encoder seeing only the visible 25% make
pretraining cheaper?

**P10 (open analysis).** DINO's [CLS]-attention segments objects with no labels; supervised ViTs' does not.
Propose a mechanistic explanation in terms of what the self-distillation-across-crops objective rewards
(view-invariant, object-centric features) versus what a supervised cross-entropy rewards (whatever is
sufficient to separate classes, possibly shortcut features). What experiment would distinguish your
explanation from the alternative that it is merely an artifact of augmentation? (Open-ended; argue carefully.)

---

## Solution sketches

**S1.** `N = (384/16)^2 = 24^2 = 576`; with CLS, 577. With `P = 32`: `N = 12^2 = 144`, a factor `144/576 =
1/4`; the `N x N` matrix shrinks by `(1/4)^2 = 1/16`.
**S2.** Output channel `k` at position `(i,j)` is `sum_{c,u,v} W[k,c,u,v] x[c, iP+u, jP+v] + b[k]`. Stride =
kernel = `P` makes the receptive fields `{iP+u}` disjoint and tiling — exactly the patches. Flatten
`W[k,:,:,:]` to row `e_k` and the patch to `x_p`; then output `= e_k · x_p + b[k]`, i.e. `E x_p + b`. The
stride=kernel condition is what makes patches non-overlapping and complete.
**S3.** `E[q·k] = sum E[q_m]E[k_m] = 0`; `Var(q·k) = sum Var(q_m k_m) = d_k` (independent unit-variance
products), so std `= sqrt(d_k)`. Large logits push softmax to near one-hot; then `p` is near a corner of the
simplex and `diag(p) - p p^T -> 0`, so gradients vanish. Dividing by `sqrt(d_k)` restores unit logit
variance and a well-conditioned softmax.
**S4.** `Q -> Pi Q`, `K -> Pi K`, `V -> Pi V`; `QK^T -> Pi QK^T Pi^T`; row-wise softmax commutes with the
symmetric permutation; times `Pi V` and using `Pi^T Pi = I` gives `Pi · softmax(QK^T/sqrt(d)) V`. So outputs
permute with inputs — no positional information survives, hence position codes are required.
**S5.** Projections + attention: `Theta(N D^2)` for Q,K,V,O plus `Theta(N^2 D)` for scores and value-mix.
MLP: two `D x 4D` matmuls per token = `Theta(N D^2)`. Attention's `N^2 D` term dominates the `N D^2` terms
when `N > D` (crossover at `N ~ D`).
**S6.** Each window has `M^2` tokens, cost `Theta(M^4 D)` per window (wait: attention within a window is
`(M^2)^2 D = M^4 D`); there are `N / M^2` windows, so total `= (N/M^2)·M^4 D = N M^2 D` — linear in `N`. At
`N=3136, M=7 (M^2=49)`: windowed `~ 3136·49·D` vs. global `3136^2·D`, a ~64x reduction. Shifting is needed
because fixed windows never exchange information across their borders.
**S7.** Reshape the `196 = 14x14` table to a `14x14xD` grid and **bicubically interpolate** to `28x28`, then
flatten to `784xD`. Truncation/padding fails because position entries are spatially structured, not a bag.
A relative position bias depends only on offset `i-j`, so it generalizes to new grid sizes without
interpolation (translation-friendly).
**S8.** `hat{A}_l = normalize(0.5 A_l + 0.5 I)`; `Rollout = hat{A}_L ... hat{A}_1`; take the CLS row. The
`0.5 I` term models the residual connection (a token retains its own information), without which rollout
overstates mixing. Rollout ignores value magnitudes and the MLP non-linearities, so it is an approximation,
not an exact attribution.
**S9.** Adjacent patches are highly correlated, so at 15% masking a missing patch is trivially inpainted from
neighbors — the model learns low-level interpolation, not semantics. ~75% masking removes enough context that
reconstruction requires modeling object/scene structure. The encoder processing only the visible 25% cuts its
sequence length ~4x, so compute (quadratic in tokens) drops sharply, enabling large models.
**S10.** Self-distillation across crops rewards features that are *invariant* to which crop/augmentation is
seen and thus latch onto the persistent object, yielding object-centric attention. Supervised CE rewards any
feature separating classes, which can be a shortcut (texture, background). To distinguish from a pure
augmentation artifact: train a supervised ViT with the *same* multi-crop augmentation and check whether its
CLS attention also segments; if not, the objective (not the augmentation) is responsible.
