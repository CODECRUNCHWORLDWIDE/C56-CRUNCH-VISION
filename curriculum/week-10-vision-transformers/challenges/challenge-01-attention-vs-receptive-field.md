# Challenge 1 — Global attention vs. local convolution, made measurable

Make the core architectural difference — global attention vs. local receptive fields — concrete and
measurable, not rhetorical.

1. Construct a task where **long-range** relationships decide the label: e.g. classify by the relationship
   between two small objects placed at opposite corners of the image (same color vs. different, or a symbol
   in one corner "pointing" toward or away from a target in the other). Locality alone cannot solve it.
2. Compare a ViT and a CNN of matched capacity, trained identically. Does the ViT's global-from-layer-one
   attention help where the CNN's local receptive field struggles? For the CNN, estimate how many stacked
   3x3 layers it would need to connect the two corners (receptive-field growth is roughly linear in depth,
   `RF ~ 1 + L·(k-1)`), and check whether your CNN is that deep.
3. Visualize the ViT's **attention rollout** on this task to show it relating the distant regions directly,
   and contrast with the CNN's feature maps.
4. Now construct the reverse: a task where *locality and translation equivariance are the whole game* (a
   texture/local-pattern classification with objects at random positions) and show the CNN's bias wins on
   small data — neither architecture is universally better.

**Deliverable:** two contrasting tasks with matched-capacity results, receptive-field arithmetic for the
CNN, and attention-rollout visualizations, demonstrating precisely where global attention wins and where
local convolution's bias wins. Grounding the abstract difference in measured tasks is the point.
