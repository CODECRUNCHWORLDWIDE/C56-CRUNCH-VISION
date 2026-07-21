# Challenge 1 — Push transfer to a different domain

Transfer is easy on ImageNet-like photos. Test its limits on a domain that doesn't resemble natural images.

1. Pick a dataset visually unlike ImageNet: medical images, satellite/aerial imagery, sketches, or microscopy (public options exist for each).
2. Apply feature extraction with a standard ImageNet backbone and record accuracy. Then fine-tune and record again.
3. Compare the *gap* between feature extraction and fine-tuning here vs. on an ImageNet-like dataset from earlier. Where does fine-tuning matter more, and why?
4. Try extracting features from an *earlier* backbone layer (more general) instead of the final one, and see if it helps on the out-of-domain data.

**Deliverable:** a results table across strategies and a paragraph on how domain distance changed what worked. The lesson — transfer helps most when domains match, and needs more adaptation when they don't — governs every applied vision decision.
