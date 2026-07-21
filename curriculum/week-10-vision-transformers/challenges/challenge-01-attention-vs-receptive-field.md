# Challenge 1 — Global attention vs. local convolution

Make the core architectural difference — global attention vs. local receptive fields — concrete and measurable.

1. Construct or find a task where **long-range** relationships matter: e.g. classifying based on the relationship between two objects at opposite corners of the image, or a texture that requires global context.
2. Compare a ViT and a CNN on it. Does the ViT's global-from-layer-one attention help where the CNN's local receptive field struggles?
3. Visualize the ViT's attention to show it relating the distant regions, and reason about how many layers a CNN would need to connect them.
4. Then find a task where *locality* is the whole game and the CNN's bias helps — showing neither is universally better.

**Deliverable:** two contrasting tasks with results and attention visualizations, demonstrating where global attention wins and where local convolution's bias wins. Grounding the abstract difference in measured tasks is the point.
