# Challenge 2 — Find what actually matters (controlled ablation)

You have many architecture knobs. Discover which ones move the needle — with evidence and error
bars, not folklore.

1. Take your CIFAR-10 CNN and systematically vary *one thing at a time*: depth (2 vs 4 conv blocks), width
   (channel counts), kernel size (3x3 stack vs one 5x5), and pooling vs. strided conv. Hold everything else
   fixed.
2. For each variant record test accuracy **and parameter count and FLOPs**. Which change gave the best
   accuracy-per-parameter? Per-FLOP?
3. Run each promising configuration with **at least 3 random seeds** and report mean +/- std. Only claim a
   difference is real if it exceeds the seed-to-seed noise.
4. Add a residual (skip) connection to your deepest variant and measure whether it improves *training*
   accuracy — a small-scale echo of the degradation experiment (Challenge 3 goes deeper).

**Deliverable:** a results table with means, standard deviations, and cost columns, plus a short analysis of
what actually improved the model *for your dataset*. Disciplined, error-barred experimentation is the
difference between engineering and cargo-culting.
