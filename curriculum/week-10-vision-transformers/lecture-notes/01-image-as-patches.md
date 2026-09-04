# Lecture 1 — An image as a sequence of patches: embedding, position, and the [CLS] token

The Transformer was built for sequences of word tokens. To apply it to images, the Vision
Transformer's central move is disarmingly simple: **chop the image into patches and treat each patch as a
token** (Dosovitskiy et al., "An Image Is Worth 16x16 Words," ICLR 2021). That reframe — image as a
sequence — is the whole conceptual leap. This lecture makes each step of the tokenizer precise, because the
"front door" is where most implementation bugs live and where the ViT's inductive biases are decided.

## Patch embedding, stated exactly

Take an image `x` of shape `(C, H, W)` — say `(3, 224, 224)`. Fix a patch size `P` (typically 16). The
image splits into a grid of `N = (H/P)(W/P) = (224/16)^2 = 196` non-overlapping patches, each of shape
`(C, P, P) = (3, 16, 16)`, i.e. `C·P^2 = 768` numbers. Flatten each patch to a vector `x_p in R^{C P^2}`
and apply a single shared linear map `E in R^{D x (C P^2)}` to get a `D`-dimensional **patch embedding**
`z_p = E x_p + b`. The image is now a sequence of `N` token vectors in `R^D`, exactly like word embeddings.

## The convolution-equals-patch-embedding theorem

There is an elegant identity worth proving, because it is the fastest correct implementation. Claim: a
**non-overlapping convolution with kernel size = stride = P** computes the patch embedding. A `Conv2d` with
`in=C, out=D, kernel=P, stride=P` places `H/P x W/P` output positions; at output position `(i, j)` it
computes, for each output channel `k`,

    y[k, i, j] = sum_{c, u, v} W[k, c, u, v] · x[c, iP + u, jP + v] + b[k].

Because stride equals kernel size, the receptive fields `{iP+u : 0<=u<P}` are disjoint and tile the image —
exactly the patches. Flatten the kernel `W[k, :, :, :]` into a row `e_k in R^{C P^2}` and flatten the patch
into `x_p`; then `y[k, i, j] = e_k · x_{(i,j)} + b[k]`, which stacked over `k` is `E x_p + b`. So the conv
*is* the per-patch linear projection, and the `(D, H/P, W/P)` output flattened over space is the token
sequence. Two worlds connect:

```python
import torch.nn as nn
patch_embed = nn.Conv2d(3, D, kernel_size=16, stride=16)   # (N,3,224,224) -> (N,D,14,14)
tokens = patch_embed(img).flatten(2).transpose(1, 2)        # (N, 196, D)
```

The equivalence is not just a trick: it tells you the ViT's *only* built-in spatial prior is the patch grid
itself — a coarse, non-overlapping 16x16 tiling. Everything finer (edges, parts, whole-object relations)
must be learned by attention, unlike a CNN whose every layer re-imposes locality.

## Positional embeddings: why, and which

Self-attention is **permutation-equivariant**: permuting the input tokens permutes the outputs identically,
with no change in values (Lecture 2 proves this). So the raw token set carries *no* notion of where a patch
sits. Patch #1 (top-left) and patch #196 (bottom-right) are interchangeable to attention. But spatial
position obviously matters — a face is not its shuffled patches. The fix is to **add** a positional code to
each token. There is a taxonomy (deepened in Lecture 4):

- **Learned absolute** (original ViT): a trainable table `P in R^{N x D}`, added to the tokens. Simple,
  effective, but tied to a fixed grid — changing input resolution requires **interpolating** the position
  table (a real deployment gotcha).
- **Sinusoidal** (Vaswani et al., 2017): fixed sines/cosines of varying frequency; extrapolates to unseen
  lengths but is 1-D by default.
- **2-D factorized**: separate row and column embeddings summed, respecting image geometry.
- **Relative position bias** (Swin; Shaw et al., 2018): add a learned bias depending on the *offset*
  between two patches, injected into the attention logits — translation-friendly.

Without any of these, "a ViT can't tell a face from its scrambled patches" is literally true.

## The class token, and its alternative

Following BERT (Devlin et al., 2019), ViT prepends a special learnable **[CLS] token** `z_cls in R^D` to the
sequence. It attends to and is attended by all patches through the stack; its final-layer output is fed to
the classification head as a learned summary of the whole image. The sequence is therefore

    [ z_cls ; z_1 + p_1 ; ... ; z_N + p_N ]   -> shape (N+1, D) = (197, D)

An equally common alternative is **global average pooling (GAP)** over the patch tokens instead of a CLS
token; Beyer et al. (2022, "Better plain ViT baselines") found GAP heads competitive and simpler. The CLS
token is a convention, not a necessity.

```mermaid
flowchart LR
  A["Image 3x224x224"] --> B["Split into 16x16 patches (N=196)"]
  B --> C["Flatten each patch to 768-vector"]
  C --> D["Shared linear projection E -> D dims"]
  D --> E["Patch embeddings z_1..z_N"]
  E --> F["Add positional embeddings p_i"]
  F --> G["Prepend learnable CLS token"]
  G --> H["Transformer encoder (12x blocks)"]
  H --> I["CLS output -> MLP classifier"]
```
*From a raw image to a token sequence a standard Transformer can process.*

## The contrast with CNNs, sharpened

A CNN builds understanding **locally and hierarchically** — small receptive fields growing with depth
(Week 3), with translation equivariance re-imposed at every layer by weight sharing. A ViT, from its first
attention layer, can relate *any* patch to *any* other: a **global receptive field immediately**. The CNN's
locality and equivariance are strong priors that make it data-efficient; the ViT has *fewer* priors (only
the patch grid), so it is more flexible but must *learn* spatial structure from data — which is why it is
data-hungry (Lecture 3). This is a genuine bias-variance trade at the architectural level.

## Common pitfalls

- **Forgetting to interpolate the position table** when you change input resolution. A ViT pretrained at
  224 with 196 position entries cannot ingest 384-px images (576 patches) without interpolating `P`.
- **Wrong preprocessing.** ViTs are sensitive to the exact normalization (mean/std) they were trained with;
  a mismatch quietly degrades accuracy. Always use the model's own transform.
- **Assuming the CLS token is special.** It is one design choice; GAP works. Do not build architecture myths
  around it.

**Takeaway:** a ViT turns an image into a sequence of patch tokens — each a linearly embedded flattened
patch (equivalently, a stride-P convolution) — plus a positional code for where it sits and, optionally, a
[CLS] token to summarize. Its only hard-wired spatial prior is the coarse patch grid; every finer structure
is learned by global self-attention, for better (flexibility, scale) and worse (data-hunger).
