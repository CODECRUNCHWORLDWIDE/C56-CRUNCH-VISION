# Exercise 3 — Close the overfitting gap

**Goal:** use the regularization and LR toolbox with evidence.

## Tasks

1. Train your classifier with *no* regularization and plot the train/validation gap — establish the overfitting baseline.
2. Add, one at a time: augmentation, weight decay, dropout. Measure each one's effect on the gap and on validation accuracy.
3. Add a learning-rate schedule (cosine or step decay) and compare final accuracy to a constant LR.
4. Add early stopping (keep the best-validation checkpoint) and confirm it prevents late-epoch overfitting.

## Deliverable

A notebook with a table showing each technique's effect on the train/val gap and accuracy, plus the with/without-schedule comparison. You should be able to say which techniques helped *your* dataset and by how much.
