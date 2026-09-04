# Lecture 5 — The architectural lineage: from LeNet to ResNet, and why depth needed residuals

A CNN is not one architecture but a two-decade design conversation, where each landmark solves a
specific failure of its predecessor. Reading the lineage this way — as answers to problems, not a list of
names — is the difference between memorizing architectures and being able to design one. This lecture traces
LeNet -> AlexNet -> VGG -> GoogLeNet -> ResNet and explains, with the degradation experiment, why residual
connections were the breakthrough that unlocked very deep networks.

## LeNet-5 (LeCun et al., 1998): the template

Two conv+pool blocks then dense layers, ~60k parameters, trained with backprop to read handwritten digits
on real bank checks. It established the *shape* the whole field still uses: convolutional feature extractor,
then a classifier head. What held the field back for a decade after was not the idea but the substrate —
data and compute.

## AlexNet (Krizhevsky, Sutskever & Hinton, 2012): scale + ReLU + the GPU moment

AlexNet won ImageNet 2012 by a stunning margin (top-5 error 16.4% vs. 26% for the runner-up) and ignited the
deep-learning era. Its contributions were pragmatic and compounding: **ReLU** activations (Nair & Hinton,
2010) that trained far faster than saturating tanh/sigmoid by not vanishing gradients on the positive side;
**dropout** in the dense head; heavy **data augmentation**; and training across two GPUs. The lesson: the
1998 template scaled — given enough data (ImageNet, Deng et al., 2009), compute, and the right nonlinearity.

## VGG (Simonyan & Zisserman, 2015): depth via stacked 3x3 convs

VGG made one disciplined argument: replace large kernels with *stacks of 3x3 convs*. Two 3x3s have the
receptive field of one 5x5 but fewer parameters (`2*9 = 18` vs. `25` per channel-pair) and an extra
nonlinearity; three 3x3s match a 7x7. Uniform 3x3 conv, 2x2 pool, repeat — depth 16-19. It is expensive
(the dense head alone holds ~120M of its ~138M parameters) but its regularity made "just go deeper with
small kernels" the default mental model.

## GoogLeNet / Inception (Szegedy et al., 2015): efficiency via factorization

Inception attacked *cost*. Its module runs 1x1, 3x3, and 5x5 convs plus pooling **in parallel** and
concatenates them, letting the network pick the scale per location. The crucial trick is **1x1 convolutions
as bottlenecks**: a 1x1 conv mixes channels and *reduces their number* before an expensive 3x3/5x5, cutting
FLOPs dramatically (the complexity formula from Lecture 4 makes this quantitative — halving `C_in` before a
3x3 halves its cost). GoogLeNet matched VGG's accuracy with ~12x fewer parameters, and replaced the giant
dense head with **global average pooling**.

## The degradation problem: deeper stopped being better

By 2015 the field hit a wall. Stacking more plain layers *increased training error* — not a test/overfitting
problem but an **optimization** failure: a 56-layer plain net had higher *training* error than a 20-layer
one (He et al., 2016, "Deep Residual Learning for Image Recognition"). This is paradoxical, because a deeper
net can represent everything a shallower one can (set the extra layers to identity) and so should do at
least as well. The optimizer simply could not *find* that solution: making a stack of nonlinear layers
learn an identity mapping is hard, and gradients through very deep plain stacks degrade.

## ResNet (He et al., 2016): learn the residual

The fix is deceptively simple. Instead of asking a block to learn a target mapping `H(x)`, ask it to learn
the **residual** `F(x) = H(x) - x` and add the input back via a **skip connection**:

    y = F(x) + x.

Now the identity mapping is trivial to represent — just drive `F` to zero, which weight decay already
encourages — so adding depth can *never* hurt the optimizer's reachable set. Just as important, the skip
gives gradients a direct additive path to every layer: differentiating `y = F(x) + x` yields
`dy/dx = dF/dx + I`, and that `+I` term prevents the gradient from vanishing no matter how deep the stack.
Residual networks trained at **152 layers** (and beyond) and won ImageNet 2015. Skip connections are now
non-negotiable in deep vision, and the same additive-shortcut idea reappears as the residual stream in the
transformers of Week 10.

## What came after (the map)

- **DenseNet** (Huang et al., 2017): connect every layer to every later one; maximal feature reuse.
- **Batch-norm-free / normalizer-free nets** and better initialization refine trainability.
- **Depthwise-separable convolutions** (MobileNets, Howard et al., 2017): factor a standard conv into a
  depthwise (one filter per channel) plus a 1x1 pointwise conv, cutting cost by ~`1/k^2 + 1/C_out` — the key
  to on-device vision (Week 11).
- **EfficientNet** (Tan & Le, 2019): *compound scaling* of depth, width, and resolution together, found by
  neural architecture search, giving a Pareto-optimal accuracy/FLOP family.
- **ConvNeXt** (Liu et al., 2022): a pure-conv net modernized with transformer-era training recipes,
  proving convolutions remain competitive with the Vision Transformers you meet in Week 10.

## Common pitfalls

- **Blaming overfitting for the degradation problem.** It was a *training-error* failure — an optimization
  problem residuals solved, not a regularization one.
- **Thinking residual = "just add layers."** The point is the identity shortcut making identity easy to
  learn and giving gradients an `+I` highway; a plain deep stack lacks both.
- **Over-parameterizing the head.** VGG's 120M-parameter dense head is the anti-pattern; global average
  pooling is almost always the right modern choice.

**Takeaway:** each landmark architecture answered a concrete failure — AlexNet showed the template scales
with data/compute/ReLU, VGG got depth cheaply via stacked 3x3s, Inception got efficiency via 1x1
bottlenecks, and ResNet solved the *degradation* (optimization) problem by learning residuals `F(x)+x`,
whose skip connections make identity trivial and hand gradients an `+I` highway — the idea that made
100+-layer vision networks trainable and that you will meet again as the residual stream in transformers.
