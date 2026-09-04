# Exercise 2 — End-to-end skeleton with a preprocessing golden test

**Goal:** the whole pipeline working, badly, early — with parity proven, not assumed.

## Tasks

1. Wire the complete path: load images → train a *tiny* model (transfer learning) for a few epochs →
   evaluate with your real metric → save → serve one prediction through a minimal `/predict` API.
2. Don't chase accuracy yet — chase *connectivity*. Every stage runs, and the served prediction uses the
   *same* preprocessing as training.
3. Add a **golden test**: pick one fixed image, run it through the training-time preprocessing + model and
   through the served pipeline, and assert the outputs match within a numerical tolerance. This catches the
   silent resize/normalize/channel-order parity bug (Week 11) before it can eat your final week.
4. Wire `eval.py` so it regenerates the headline metric from saved artifacts with one command.

## Deliverable

A repo where `train.py` produces a saved model, `eval.py` regenerates the metric, the serving script returns
a prediction on a real image, and the golden test passes — even if the model is weak. This de-risks the
entire capstone.
