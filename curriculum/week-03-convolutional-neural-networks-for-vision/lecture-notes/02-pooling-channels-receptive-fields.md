# Lecture 2 — Pooling, channels, and the receptive field (theoretical vs. effective)

A conv layer sees only a small neighborhood. To recognize a whole object, the network must
integrate information across the image — it does this by getting *deeper* and *coarser*. This lecture is
the anatomy of a CNN, with the receptive-field arithmetic made exact and a surprising fact: the region a
deep neuron *actually* uses is far smaller than the region it *could* use.

## Channels are parallel feature detectors

Each output channel is one **feature map** — one learned filter's response across the image. Early channels
fire on edges and colors; deeper channels fire on textures, then parts, then objects (visualized directly
by Zeiler & Fergus, 2014, "Visualizing and Understanding Convolutional Networks"). "64 channels" means
"64 learned detectors run in parallel." Depth trades *where* for *what*: spatial size shrinks while channel
count grows.

## Output-shape arithmetic

For one spatial dimension with input size `W`, kernel `k`, padding `p`, stride `s`, dilation `d`, the
output size is

    W_out = floor( (W + 2p - d*(k-1) - 1) / s ) + 1.

Memorize this; it prevents most CNN bugs. "Same" padding (output size = input size at stride 1) needs
`p = (k-1)/2` for odd `k` — the reason kernels are almost always odd. **Dilation** `d` inflates the kernel's
footprint without adding weights (atrous/dilated convolution, Yu & Koltun, 2016), inserting `d-1` gaps
between taps — a cheap way to enlarge the receptive field, central to segmentation (Week 7).

## Pooling: shrink space, keep signal

**Pooling** downsamples a feature map. **Max pooling** with a 2x2 window, stride 2 keeps the strongest
response per block, halving H and W:

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)   # (N,C,32,32) -> (N,C,16,16)
```

Why pool? (1) it cuts computation as depth grows; (2) it grows the receptive field fast; (3) it adds a
little *local* translation invariance — the block max is unchanged if the feature shifts within the block.
Modern nets often replace pooling with **strided convolution** (a stride-2 conv both filters and
downsamples; Springenberg et al., 2015, "Striving for Simplicity: The All Convolutional Net"), and end
with **global average pooling** (Lin et al., 2014, "Network in Network") instead of a giant flatten-then-
dense, which slashes parameters and improves generalization.

## The theoretical receptive field: the recurrence

The **receptive field** (RF) of a neuron is the region of the *input image* that can influence it. It grows
predictably. Track the RF size `r` and the cumulative stride (jump) `j`. Initialize `r_0 = 1`, `j_0 = 1`.
For a layer with kernel `k` and stride `s`:

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in

Worked example: two stacked `3x3` stride-1 convs. Layer 1: `r = 1 + 2*1 = 3`, `j = 1`. Layer 2:
`r = 3 + 2*1 = 5`, `j = 1`. So two `3x3` convs see `5x5` — with `2*(3*3) = 18` weights per channel-pair
versus one `5x5` kernel's `25`, plus an extra nonlinearity. Three `3x3`s reach `7x7`. **Stacking small
convs to grow the receptive field cheaply and with more nonlinearity is the core VGG insight** (Simonyan &
Zisserman, 2015). Add a pooling/stride-2 layer and `j` doubles, so the RF then grows in strides of 2, 4,
8... — the field balloons and deep neurons soon "see" the whole image, which is exactly what a global
judgment ("this is a cat") requires.

## The effective receptive field: the twist

Here is the fact introductory courses omit. The *theoretical* RF is an upper bound on what a neuron *could*
see. Luo et al. (2016, "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks")
showed that the **effective** receptive field — where the gradient magnitude is actually non-negligible —
is (i) roughly **Gaussian**, concentrated at the center, and (ii) far *smaller* than the theoretical RF,
scaling only like `O(sqrt(#layers))` rather than linearly. Intuitively, central pixels have many more paths
to the output neuron than edge pixels (a random-walk / central-limit effect), so their influence dominates.

Practical consequences: (1) simply stacking more layers grows the *usable* field sublinearly, motivating
dilation, larger strides, and — later — attention (Week 10) for genuinely global context; (2) padding and
architecture choices that fight the center bias matter for dense-prediction tasks. This is a live research
theme, not a settled footnote.

## The classic CNN shape

```
[Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool]  (repeat: channels up, spatial down)
-> Global average pool
-> Dense -> softmax over classes
```

Spatial size shrinks (32->16->8->4) while channels grow (3->64->128->256). You end with a small grid of
rich features summarizing the whole image, which a light head maps to class scores. This "convolutional
feature extractor, then a classifier head" template is the backbone of the entire course.

## Common pitfalls

- **Trusting the theoretical RF.** Your deep neuron *can* see the whole image but *effectively* attends to a
  small central blob. If global context matters, dilation/attention — not just depth — is the fix.
- **Even kernels.** They make "same" padding asymmetric; use odd kernels unless you have a reason.
- **Flattening a large feature map into a dense head.** It reintroduces the parameter explosion; prefer
  global average pooling.

**Takeaway:** channels are parallel learned detectors; pooling and stride shrink space while the receptive
field grows by the recurrence `r += (k-1)*j`, `j *= s`. But the *effective* RF is Gaussian and only
`O(sqrt(depth))` wide — deep neurons could see everything yet mostly attend to the center, which is why
dilation and later attention exist.
