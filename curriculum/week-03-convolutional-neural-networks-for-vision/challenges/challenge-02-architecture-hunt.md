# Challenge 2 — Find what actually matters

You have many architecture knobs. Discover which ones move the needle — with evidence, not folklore.

1. Take your CIFAR-10 CNN and systematically vary *one thing at a time*: depth (2 vs 4 conv blocks), width (channel counts), kernel size (3×3 vs 5×5), and pooling vs. strided conv. Hold everything else fixed.
2. For each, record test accuracy and parameter count. Which change gave the best accuracy-per-parameter?
3. Add batch normalization after convs and measure its effect on training speed and final accuracy.
4. Write up which knobs mattered most for *your* dataset and which were noise.

**Deliverable:** a results table and a short analysis of what actually improved the model. Resist over-claiming from a single run — note where differences are within run-to-run noise. Disciplined experimentation is the difference between engineering and cargo-culting.
