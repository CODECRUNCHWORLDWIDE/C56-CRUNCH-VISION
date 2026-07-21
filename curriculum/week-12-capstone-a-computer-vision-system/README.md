# Week 12 — Capstone: a computer-vision system

> **Goal:** by Sunday you have designed, built, trained, honestly evaluated, and deployed one computer-vision system on a problem you chose, beating a documented baseline and packaged as a repo someone else can clone, run on their own images, and trust.

This is where it all comes together. No new theory — the capstone asks you to run the entire pipeline you've practiced for eleven weeks on a problem *you* pick: frame it (classification, detection, or segmentation), build and train a model, beat a baseline, evaluate honestly with an error analysis and named failure modes, then export and serve it behind an inference API with a model card. The deliverable is not a notebook — it's a computer-vision system you'd put in front of an employer, and defend.

## Learning objectives

By the end of this week, you will be able to:

- **Frame** a real vision problem as a classification, detection, or segmentation task with a clear metric and a documented baseline.
- **Build and train** an appropriate model, applying transfer learning and the training toolbox deliberately.
- **Evaluate** honestly on held-out images with an error analysis, dataset-bias check, and named failure modes.
- **Deploy** the model behind an inference API (optimized for the target if edge), with verified preprocessing parity.
- **Communicate** the work: a clear README, a model card, and an honest limitations and ethics section.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Scoping a vision project that finishes](lecture-notes/01-scoping-a-vision-project.md)
2. [Lecture 2 — Honest evaluation & error analysis for vision](lecture-notes/02-honest-vision-evaluation.md)
3. [Lecture 3 — Deploying & communicating a vision system](lecture-notes/03-deploy-and-communicate.md)

**Exercises**

1. [Exercise 1 — Write the proposal and baseline](exercises/exercise-01-proposal-and-baseline.md)
2. [Exercise 2 — End-to-end skeleton](exercises/exercise-02-end-to-end-skeleton.md)
3. [Exercise 3 — Iterate, then evaluate honestly](exercises/exercise-03-iterate-and-evaluate.md)

**Challenges**

1. [Challenge 1 — Beat the baseline, provably](challenges/challenge-01-beat-your-baseline.md)
2. [Challenge 2 — Demo and defend](challenges/challenge-02-demo-and-defend.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 12.

## Deliverable

A polished, reproducible repository: problem framing, a trained vision model beating a baseline on held-out images, honest evaluation with error analysis and a bias check, a served inference API with preprocessing parity, a model card, and a README anyone can follow to reproduce your results on their own images.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
