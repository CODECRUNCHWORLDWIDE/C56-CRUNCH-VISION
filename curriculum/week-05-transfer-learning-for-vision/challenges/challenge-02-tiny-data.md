# Challenge 2 — How little data can you win with?

Transfer learning's headline promise is winning with little data. Quantify exactly how little, and
find where the modern backbones move the frontier.

1. Take a dataset and train your best transfer model with progressively fewer images per class: e.g. 100, 50,
   20, 10, 5, and (stretch) 1-shot.
2. Plot accuracy vs. training-set size (log x-axis). Where does performance start to collapse?
3. At the smallest sizes, show whether heavy augmentation and **feature extraction / linear probing** (vs. full
   fine-tuning) extend how far you can go — the small-data corner of the Lecture-2 grid, and where LP/PEFT
   should win.
4. Compare the whole curve to a **from-scratch CNN's** — the from-scratch model should collapse far sooner.
5. **Frontier move.** Repeat the smallest-data points with a DINO/CLIP backbone (Lecture 5) and, for CLIP, add
   the **zero-shot** point (no target labels at all). Does a stronger pretraining source lower the
   minimum-viable dataset?

**Deliverable:** an accuracy-vs-data-size plot for transfer (multiple backbones and strategies) vs. from-scratch,
with CLIP zero-shot marked, and a short analysis of the minimum viable dataset for your task. Knowing how much
data you *actually* need is a hugely practical, money-saving judgment.
