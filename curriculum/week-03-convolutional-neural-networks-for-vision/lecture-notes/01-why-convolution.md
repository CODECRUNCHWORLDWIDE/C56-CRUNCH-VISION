# Lecture 1 — Why convolution, not fully-connected: equivariance forces the operator

You could flatten a 28x28 image into a 784-vector and feed it to a fully-connected (dense) layer.
For tiny images it even works. But it is the *wrong* architecture, and the reason is not merely "too many
parameters" — it is that a dense layer throws away the one structural fact that makes images learnable.
This lecture proves that convolution is not a clever heuristic but the *forced* answer to a symmetry
requirement.

## What is wrong with a dense layer, precisely

Three failures, in increasing order of importance.

- **Parameter explosion.** A dense map from a 224x224x3 image (150,528 inputs) to 1,000 hidden units needs
  ~150 million weights in *one* layer. Un-trainable on realistic data budgets.
- **No spatial prior.** Flattening destroys the grid: the network does not know pixel (10,10) neighbors
  (10,11). It must *re-learn* adjacency from data, wasting capacity on a fact we already know.
- **No translation invariance.** A digit in the top-left and the same digit in the bottom-right are, to a
  dense net, unrelated input patterns. It would have to learn "7" separately at every location — an
  exponential waste.

The third failure is the deep one. Fix it correctly and the other two fix themselves.

## Equivariance, stated exactly

Let `T_v` denote translation of an image by vector `v`: `(T_v x)[p] = x[p - v]`. A layer `f` is
**translation-equivariant** if shifting the input shifts the output identically:

    f(T_v x) = T_v f(x)   for all shifts v.

This is the mathematical statement of "a feature detector should fire wherever the feature is." Now the
theorem (a standard result in harmonic analysis; see Mallat, *A Wavelet Tour of Signal Processing*, 2009,
and Cohen & Welling, 2016, "Group Equivariant Convolutional Networks"):

> **A linear map on a grid is translation-equivariant if and only if it is a convolution.**

Sketch: a linear map is `y = Wx`. Equivariance `W T_v = T_v W` for every shift `v` forces the matrix `W` to
be **circulant** (each row a shifted copy of the one above). A circulant matrix is exactly the matrix form
of convolution with a single kernel. So the moment you demand equivariance, you have *derived* weight
sharing — the kernel is shared across positions because equivariance says it must be. Add the further
requirement of **locality** (each output depends only on a bounded neighborhood) and the kernel has compact
support: a small `k x k` window.

So convolution is not chosen for convenience. Given the symmetry of natural images, it is the *unique*
linear layer consistent with that symmetry. Everything else — few parameters, spatial prior — is a
downstream consequence.

## A convolution layer, precisely

A conv layer holds `C_out` kernels, each of shape `(C_in, k, k)`. Each kernel convolves over all input
channels and produces one output **feature map** (one channel). The layer maps a volume `(C_in, H, W)` to
`(C_out, H', W')`. Learnable parameters: `C_out * C_in * k * k` weights plus `C_out` biases — independent
of `H` and `W`.

Parameter comparison, made concrete. A `3x3` conv from 3 to 64 channels on any image: `3*64*3*3 + 64 =
1,792` weights. A dense layer producing the same `64 x 224 x 224` output from the same input would need
~`150,528 * 3,211,264 ~= 4.8 x 10^11` weights. The conv is smaller by eight orders of magnitude, and it
*generalizes better* because its inductive bias matches the data.

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
# input (N, 3, 224, 224) -> output (N, 64, 224, 224); 1,792 learnable weights.
```

## Equivariance vs. invariance

Convolution is *equivariant*: move the cat, the "cat-ear" feature map lights up in a new place. But a
classifier needs *invariance*: "cat" regardless of position. The network converts equivariance into
invariance gradually — through pooling and, at the end, global pooling that discards position. Hold this
distinction: conv layers are equivariant; the *head* is what makes the whole network invariant.

## The biological and historical thread

The architecture was not invented in a vacuum. Hubel & Wiesel's Nobel-winning recordings of cat visual
cortex (1962) found *simple cells* responding to oriented edges at specific locations and *complex cells*
pooling over position — a literal edge-detector-then-pool motif. Fukushima's Neocognitron (1980) encoded
this directly, and LeCun et al. (1998, "Gradient-based learning applied to document recognition")
turned it into a backprop-trainable CNN, LeNet-5, that read handwritten checks at scale.

## Week 1, learned

The operation is *exactly* the convolution you hand-coded in Week 1 — slide a kernel, weighted sum. (Note:
frameworks implement cross-correlation, i.e. no kernel flip; because the weights are learned, the flip is
absorbed and the distinction is immaterial for learning, though it matters when you derive the backward
pass in Lecture 4.) The only change: backprop *learns* the weights that minimize the loss. The first layer
reliably learns oriented edge and color-blob detectors that look strikingly like the Gabor and Sobel
filters engineers hand-designed for decades. The network rediscovers classical vision on its own.

## Common pitfalls

- **"It's just fewer parameters."** No — it is the *right* parameters. A randomly-pruned dense net with the
  same parameter count does far worse; the win is the equivariant structure, not the sparsity.
- **Confusing equivariance with invariance.** The conv stack is equivariant; invariance is manufactured by
  pooling and the head. Losing this distinction makes segmentation (Week 7) confusing later.
- **Ignoring cross-correlation vs. convolution.** Fine for the forward pass with learned weights; it will
  bite you in the backward-pass derivation if you are sloppy.

**Takeaway:** convolution is the *unique* translation-equivariant local linear layer — weight sharing is a
theorem, not a hack. That structural prior is why a 1,792-weight conv beats a 10^11-weight dense layer on
images: it encodes the symmetry of the visual world instead of relearning it from scratch.
