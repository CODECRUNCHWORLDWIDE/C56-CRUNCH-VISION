# Week 9 — Video understanding & optical flow

> **Goal:** by Sunday you can explain and estimate optical flow, describe how video models add a temporal dimension to vision (3D CNNs, two-stream, and video Transformers), and run an action-recognition model on short clips while reasoning about its cost and failure modes.

Video is images plus **time**, and time carries information a single frame never can — a person 'sitting' vs. 'standing up' looks identical in one frame and obvious across several. This week covers the two pillars of video understanding: **optical flow** (the per-pixel motion field between frames) and **temporal modeling** (architectures that reason across frames — 3D CNNs, two-stream networks, and video Transformers). You'll estimate flow, run an action recognizer, and confront the real costs of video: compute, memory, and data.

## Learning objectives

By the end of this week, you will be able to:

- **Define** optical flow and estimate it, understanding the brightness-constancy assumption and its limits.
- **Explain** why temporal context is essential and how it changes what tasks become possible.
- **Compare** video architectures — frame aggregation, 3D CNNs, two-stream, and video Transformers.
- **Run** a pretrained action-recognition model on short clips and interpret its predictions.
- **Reason** about the compute, memory, and data costs that make video uniquely expensive.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Optical flow: the motion field](lecture-notes/01-optical-flow.md)
2. [Lecture 2 — Architectures for video understanding](lecture-notes/02-video-architectures.md)
3. [Lecture 3 — Video in practice: cost, data & honesty](lecture-notes/03-video-in-practice.md)

**Exercises**

1. [Exercise 1 — Estimate and visualize optical flow](exercises/exercise-01-optical-flow.md)
2. [Exercise 2 — Run an action recognizer](exercises/exercise-02-run-action-recognition.md)
3. [Exercise 3 — The single-frame baseline](exercises/exercise-03-single-frame-baseline.md)

**Challenges**

1. [Challenge 1 — Build something with optical flow](challenges/challenge-01-flow-application.md)
2. [Challenge 2 — Map the video cost frontier](challenges/challenge-02-video-cost-frontier.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 10.

## Deliverable

A notebook that estimates and visualizes optical flow on a short clip (color-coded motion field), and runs a pretrained action-recognition model on a few clips, reporting predictions and an honest note on cost and failure modes.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
