# Challenge 1 — Push transfer to a different domain, and explain the gap with theory

Transfer is easy on ImageNet-like photos. Test its limits on a domain that does not resemble natural
images, and explain what you see with Lecture 4.

1. Pick a dataset visually unlike ImageNet: medical images, satellite/aerial imagery, sketches, or microscopy
   (public options exist for each). State the low-level (texture/colour) and high-level (semantic) distance
   from ImageNet.
2. Apply feature extraction with a standard ImageNet backbone and record accuracy. Then fine-tune (responsibly:
   warmup, small LR, layer-wise decay) and record again.
3. Compare the *gap* between feature extraction and fine-tuning here vs. on an ImageNet-like dataset from
   earlier. Where does fine-tuning matter more, and why — connect the gap to the `d_{H Delta H}` divergence term.
4. Try extracting features from an *earlier* backbone layer (more general, lower divergence) instead of the
   final one, and see if it helps on the out-of-domain data. Relate to the general-to-specific hierarchy.
5. **Backbone swap.** Repeat step 2 with a self-supervised or CLIP backbone (Lecture 5). Does a more general
   pretraining source narrow the domain gap?

**Deliverable:** a results table across strategies, depths, and backbones, plus a paragraph on how domain
distance changed what worked, written in the source-error / divergence / adaptability language of the
adaptation bound. The lesson — transfer helps most when domains match and needs more (or better) adaptation
when they do not — governs every applied vision decision.
