# Challenge 2 — How little data can you win with?

Transfer learning's headline promise is winning with little data. Quantify exactly how little.

1. Take a dataset and train your best transfer model with progressively fewer images per class: e.g. 100, 50, 20, 10, 5.
2. Plot accuracy vs. training-set size. Where does performance start to collapse?
3. At the smallest sizes, show whether heavy augmentation and feature extraction (vs. fine-tuning) extend how far you can go.
4. Compare the whole curve to a from-scratch CNN's — the from-scratch model should collapse far sooner.

**Deliverable:** an accuracy-vs-data-size plot for transfer vs. from-scratch, and a short analysis of the minimum viable dataset for your task. Knowing how much data you *actually* need is a hugely practical, money-saving judgment.
