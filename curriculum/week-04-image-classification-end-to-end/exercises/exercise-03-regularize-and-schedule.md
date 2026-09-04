# Exercise 3 — Close the overfitting gap, with evidence

**Goal:** use the regularization, optimizer, and LR toolbox deliberately and measure each effect.

## Tasks

1. Train your classifier with *no* regularization and plot the train/validation gap — establish the overfitting
   baseline.
2. Add, one at a time, and measure each one's effect on the gap and on validation accuracy: augmentation, weight
   decay, dropout, label smoothing. Keep a table.
3. **Optimizer + schedule.** Compare (a) constant LR vs. cosine annealing, and (b) SGD+momentum vs. AdamW, on
   final accuracy. Note that AdamW (decoupled decay) is the correct way to combine Adam with weight decay.
4. Add early stopping (keep the best-validation checkpoint) and confirm it prevents late-epoch overfitting.
5. Add a short **warmup** to the schedule and report whether it stabilizes the first few epochs (especially if
   you increase the batch size).

## Deliverable

A notebook with a table showing each technique's effect on the train/val gap and accuracy, plus the
constant-vs-cosine and SGD-vs-AdamW comparisons. You should be able to say which techniques helped *your*
dataset and by how much — and which were within the noise of Exercise 2's confidence interval.
