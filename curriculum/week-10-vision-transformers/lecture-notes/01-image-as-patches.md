# Lecture 1 — An image as a sequence of patches

The Transformer was built for sequences of word tokens. To use it on images, the Vision Transformer's key move is disarmingly simple: **chop the image into patches and treat each patch as a token.** That reframe — image as a sequence — is the whole conceptual leap.

## Patch embedding

Take a 224×224 image and cut it into a grid of fixed-size patches, say 16×16 pixels. That's `(224/16)² = 196` patches. Each patch (16×16×3 = 768 pixels) is flattened and passed through a single linear layer to produce a **patch embedding** — a vector, exactly like a word embedding. The image is now a sequence of 196 token vectors.

```python
import torch.nn as nn
# A patch embedding is literally a conv with kernel=stride=patch_size:
patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)  # (N,3,224,224)->(N,dim,14,14)
tokens = patch_embed(img).flatten(2).transpose(1, 2)         # (N, 196, dim)
```

Note the elegant trick: a non-overlapping convolution with stride = kernel size *is* patch embedding. The two worlds connect.

## Positional embeddings

A Transformer's attention is **permutation-invariant** — it has no built-in notion of order or 2-D position. But patch #1 (top-left) and patch #196 (bottom-right) are spatially different, and that matters. So we *add* a learned **positional embedding** to each patch token, encoding where it sits in the grid. Without positional embeddings, a ViT can't tell a face from its scrambled patches.

## The class token

Following BERT, ViT prepends a special learnable **[CLS] token** to the sequence. After the Transformer layers, this token's output vector summarizes the whole image and is fed to a classification head. (Alternatively, some ViTs average all patch tokens — global average pooling — instead.)

So the input sequence is: `[CLS] + 196 patch embeddings + positional embeddings`, and it flows into a standard Transformer encoder.

```mermaid
flowchart LR
  A["Image 224 by 224"] --> B["Split into 16 by 16 patches"]
  B --> C["Flatten each patch"]
  C --> D["Linear projection"]
  D --> E["Patch embeddings"]
  E --> F["Add positional embeddings"]
  F --> G["Prepend CLS token"]
  G --> H["Transformer encoder"]
  H --> I["CLS output to classifier"]
```
*From a raw image to a sequence of tokens a Transformer can process.*

## The contrast with CNNs

A CNN builds understanding *locally and hierarchically* — small receptive fields growing with depth (Week 3). A ViT, from its very first attention layer, can relate *any* patch to *any* other — a **global receptive field immediately**. A CNN has strong built-in assumptions (locality, translation equivariance) that make it data-efficient; a ViT has *fewer* built-in assumptions, so it's more flexible but must *learn* those spatial priors from data — which is why it's data-hungry (next lecture).

**Takeaway:** a Vision Transformer turns an image into a sequence of patch tokens (a patch is a 'word'), each a linearly-embedded flattened patch, plus a learned positional embedding for where it sits and a [CLS] token to summarize. This reframing lets the Transformer's global self-attention operate on images — with far fewer built-in spatial assumptions than a CNN, for better and worse.
