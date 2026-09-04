# Week 10 — Vision Transformers

> **Goal:** by Sunday you can (1) derive the patch-embedding pipeline and prove that a non-overlapping stride-P convolution is exactly a per-patch linear projection, and classify positional-encoding schemes (learned, sinusoidal, relative, RoPE, 2-D interpolated) by the invariance each imposes; (2) write scaled dot-product and multi-head attention from the definition, count its FLOPs and memory as a function of token count N and dimension d, and explain why the softmax temperature 1/sqrt(d) is not decorative; (3) state precisely why a ViT is data-hungry — fewer hard-wired symmetries than a CNN — and predict from the ViT, DeiT, and scaling-law papers when it wins; and (4) explain the modern ViT ecosystem (Swin's shifted windows, MAE/DINO self-supervision, CLIP/SAM) at the level of what each pretraining objective actually optimizes — then build and benchmark the pipeline yourself, with measurement rather than fashion.

For a decade, convolution *was* vision. Then the Transformer — Vaswani et al.'s *Attention Is All You Need* (NeurIPS 2017), which had already displaced recurrence in language — arrived for images. Dosovitskiy et al.'s *An Image Is Worth 16x16 Words* (ICLR 2021) made the reframe that defines this week: cut an image into a grid of patches, linearly embed each into a token, add a position code, and feed the sequence to a **standard Transformer encoder**. No convolution, no pyramid, no translation-equivariance baked in — just **self-attention** relating every patch to every other from the very first layer.

That reframe is deceptively small and enormously consequential. A CNN carries strong **inductive biases** — locality and translation equivariance — that make it data-efficient but also constrain it. A ViT throws almost all of them away and *learns* spatial structure from data, which is why the original ViT **underperforms** a comparable ResNet when trained from scratch on ImageNet-1k yet **matches or beats** it after pretraining on JFT-300M. Understanding exactly which symmetries you are trading away, what self-attention costs (it is quadratic in the number of patches, a single fact that drives the entire efficient-attention literature), and when the trade pays off is the mark of a current practitioner — not the ability to recite 'Transformers replaced CNNs,' which is a headline, not an engineering fact (ConvNeXt, Liu et al. 2022, showed a modernized pure CNN matches ViTs).

This week goes past 'run a pretrained ViT.' You will derive attention and its complexity, dissect positional encodings, read the scaling laws, and reach the frontier: **self-supervised ViTs** (MAE's masked-patch reconstruction, He et al. 2022; DINO's self-distillation, Caron et al. 2021, whose attention maps segment objects with no labels) and **multimodal ViTs** (CLIP, SAM). Your C53 Transformer knowledge carries over directly; here it meets pixels.

## Learning objectives

By the end of this week, you will be able to:

- **Derive** the patch-embedding pipeline and prove a non-overlapping stride-P convolution equals a flatten-then-linear projection, then account for every token in the `[CLS] + N patch + position` sequence.
- **Classify** positional-encoding schemes — learned absolute, sinusoidal, 2-D factorized, relative-bias, and rotary — by the symmetry each imposes, and explain why a ViT without position codes cannot distinguish an image from its shuffled patches.
- **Implement** scaled dot-product and multi-head self-attention from the definition, and justify the `1/sqrt(d)` temperature by the variance of the dot product.
- **Analyze** attention's time and memory cost as Theta(N^2 d), predict how halving the patch size raises cost 16x, and derive the linear-cost trick behind FlashAttention and windowed attention.
- **Judge** honestly when ViTs beat CNNs — from the ViT, DeiT, and scaling-law evidence — reasoning explicitly about inductive bias, data scale, transfer, resolution, and edge deployment.
- **Explain** the efficient/hierarchical zoo (Swin shifted windows, PVT/pyramid downsampling, linear attention) and how each restores locality or breaks the N^2 barrier.
- **Contrast** the modern pretraining objectives — supervised, DeiT distillation, MAE masked reconstruction, DINO self-distillation, CLIP contrastive — by what each actually optimizes and what representation it yields.
- **Build** and benchmark the full pipeline: patch embedding from scratch, a pretrained ViT with attention-rollout visualization, and a measured ViT-vs-CNN comparison including the from-scratch small-data collapse.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 231N` — explain and apply attention-based vision models, and compare them against convolutional ones on evidence. |
| Industry | Choose between a transformer and a convolutional model for a real dataset on measured accuracy, wall-clock time and overfitting gap, rather than on which is newer. |
| Beyond the bar | derives attention's quadratic-in-tokens cost and makes the learner measure how halving the patch size moves it — `exercises/exercise-04-attention-cost-and-windows.md` |

## Prerequisites

- The Transformer from C53 (Crunch Nets): scaled dot-product attention, multi-head attention, residual + LayerNorm blocks, and the training tricks that keep deep stacks stable.
- CNNs and receptive fields (C56 Week 3) and transfer learning / fine-tuning (Week 5).
- Linear algebra: matrix products, softmax, eigen-intuition; basic probability (variance of a sum of independent products) for the sqrt(d) argument.
- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — An image as a sequence of patches: embedding, position, and the [CLS] token](lecture-notes/01-image-as-patches.md)
2. [Lecture 2 — Self-attention over patches: the algebra, the sqrt(d), and the N^2 cost](lecture-notes/02-self-attention-for-vision.md)
3. [Lecture 3 — ViT vs. CNN: inductive bias, data-hunger, and the scaling laws](lecture-notes/03-vit-vs-cnn.md)
4. [Lecture 4 — The geometry of attention and the efficient/hierarchical zoo](lecture-notes/04-attention-geometry-and-efficient-transformers.md)
5. [Lecture 5 — Self-supervised and multimodal ViTs: MAE, DINO, CLIP, and SAM](lecture-notes/05-self-supervised-and-multimodal-vits.md)

**Exercises**

1. [Exercise 1 — Patch embedding from scratch and the conv equivalence](exercises/exercise-01-patch-embedding.md)
2. [Exercise 2 — Run a pretrained ViT and compute attention rollout](exercises/exercise-02-run-a-vit.md)
3. [Exercise 3 — ViT vs. CNN under a controlled data budget](exercises/exercise-03-vit-vs-cnn-small-data.md)
4. [Exercise 4 — Measure the N^2 cost and implement windowed attention](exercises/exercise-04-attention-cost-and-windows.md)

**Challenges**

1. [Challenge 1 — Global attention vs. local convolution, made measurable](challenges/challenge-01-attention-vs-receptive-field.md)
2. [Challenge 2 — Interrogate the hype: architecture vs. recipe vs. cost](challenges/challenge-02-efficiency-and-hype.md)
3. [Challenge 3 — Probe a self-supervised ViT (DINO/MAE) and read its emergent structure](challenges/challenge-03-self-supervised-vit-probe.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 11.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook that: implements patch embedding two ways (flatten+Linear and stride-P Conv2d) and asserts they match; assembles the `[CLS] + N + position` sequence; runs a pretrained ViT with correct preprocessing and overlays an attention-rollout map on the object; fine-tunes a ViT and a comparable CNN on a small dataset and reports accuracy, wall-clock, and the overfitting gap; reproduces the from-scratch collapse of the ViT on limited data; and closes with a measured, hype-free written analysis tying every number to inductive bias, data scale, and the N^2 cost.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
