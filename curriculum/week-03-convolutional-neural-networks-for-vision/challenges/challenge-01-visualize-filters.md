# Challenge 1 — See what your network learned

A trained CNN is not a black box — you can look inside it. Visualize what your layers respond to
and connect it to classical vision.

1. Extract the **first conv layer's learned kernels** and display them as small images (for a 3-channel
   input they are RGB patches). Do any resemble oriented-edge or color-opponent detectors — the Gabor/Sobel-
   like filters engineers hand-designed, and the simple cells Hubel & Wiesel recorded?
2. Pick an input image and display several **feature maps** (post-activation) from the first and a deeper
   conv layer. Describe how early maps respond to edges/textures and deeper maps to more abstract parts.
3. Go further: implement a simple **activation-maximization** for one deeper channel — start from noise and
   gradient-*ascend* the input to maximize that channel's mean activation (Erhan et al., 2009). What does the
   channel "want" to see? Relate to Zeiler & Fergus (2014).
4. Argue what the network *gained* by learning these detectors rather than having them hand-engineered
   (Week 2), and one thing it *lost* (e.g. interpretability, guarantees).

**Deliverable:** a figure of first-layer filters, a few feature maps, and one activation-maximization image,
with a paragraph interpreting them. Being able to look inside a model and reason about it is a core
professional skill — and an honesty check.
