# Challenge 2 — When would you *not* use deep learning?

Deep learning is not always the answer. Build the argument for classical vision with evidence.

1. Pick a concrete task where classical features plausibly win: real-time panorama stitching on a phone, matching two images with *no* labeled training data, or localizing a robot on a CPU with a tight latency budget.
2. Time your ORB match pipeline on an image pair and estimate its speed. Contrast with what training and running a deep feature-matcher would require (data, compute, latency, interpretability).
3. Write a short decision guide: given constraints (data availability, latency, hardware, need for explainability, task difficulty), when do you reach for classical features vs. a neural network?

**Deliverable:** a one-page decision guide backed by your timing numbers. The mark of an engineer is choosing the *right* tool honestly, not defaulting to the fashionable one.
