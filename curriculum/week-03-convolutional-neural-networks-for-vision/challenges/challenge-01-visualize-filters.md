# Challenge 1 — See what your network learned

A trained CNN is not a black box — you can look inside it. Visualize what your first layers respond to.

1. Extract the **first conv layer's learned kernels** and display them as small images (for a 3-channel input, they're RGB patches). Do any resemble edge or color detectors — the Gabor/Sobel-like filters engineers used to hand-design?
2. Pick an input image and display several **feature maps** from the first and a deeper conv layer (the activation after the conv). Describe how early maps respond to edges/textures and deeper maps to more abstract structure.
3. Connect this back to Week 2: the network *learned* feature detectors that classical vision *engineered*. Argue what the network gained by learning them.

**Deliverable:** a figure of first-layer filters and a few feature maps, with a paragraph interpreting them. Being able to look inside a model and reason about it is a core professional skill — and an honesty check.
