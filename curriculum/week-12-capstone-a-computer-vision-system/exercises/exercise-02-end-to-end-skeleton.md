# Exercise 2 — End-to-end skeleton

**Goal:** the whole pipeline working, badly, early.

## Tasks

1. Wire the complete path: load images → train a *tiny* model (transfer learning) for a few epochs → evaluate → save → serve one prediction through a minimal API.
2. Don't chase accuracy yet — chase *connectivity*. Every stage runs, and the served prediction uses the *same* preprocessing as training.
3. Commit this skeleton. Now every later improvement drops into a working system.

## Deliverable

A repo where `train.py` produces a saved model and the serving script returns a prediction on a real image — even if the model is weak. This de-risks the entire capstone, especially the preprocessing-parity trap.
