# Week 2 — Classical CV: edges, features, descriptors

> **Goal:** by Sunday you can run a full Canny edge pipeline, detect corners and keypoints, describe them with a descriptor, and match features between two photos of the same scene — understanding why these classical methods still ship in production alongside deep nets.

Before deep learning, computer vision was **hand-engineered features**, and a surprising amount of it still runs today — in your phone's panorama stitching, in visual SLAM on robots, in fast on-device tracking. This week you build the classical pipeline: **edges** (Canny), **corners** (Harris), **keypoints and descriptors** (ORB/SIFT), and **matching** two views of a scene. It matters not as nostalgia but because these methods are fast, need no training data, and teach you what a 'feature' even is — the concept a CNN later learns instead of being told.

## Learning objectives

By the end of this week, you will be able to:

- **Build** a Canny edge detector and explain each of its stages.
- **Detect** corners with the Harris response and explain why corners are good features.
- **Extract** keypoints and descriptors (ORB or SIFT) and explain what a descriptor encodes.
- **Match** features between two images and reject bad matches with a ratio test.
- **Judge** when a classical method beats a neural one — speed, data, interpretability, and no-training constraints.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Edges and the Canny pipeline](lecture-notes/01-edges-and-canny.md)
2. [Lecture 2 — Corners, keypoints, and why they matter](lecture-notes/02-corners-and-keypoints.md)
3. [Lecture 3 — Descriptors and matching two images](lecture-notes/03-descriptors-and-matching.md)

**Exercises**

1. [Exercise 1 — Build and tune a Canny edge detector](exercises/exercise-01-canny-pipeline.md)
2. [Exercise 2 — Detect corners and keypoints](exercises/exercise-02-corners-and-keypoints.md)
3. [Exercise 3 — Match features between two views](exercises/exercise-03-match-two-images.md)

**Challenges**

1. [Challenge 1 — Stitch a panorama](challenges/challenge-01-panorama-stitch.md)
2. [Challenge 2 — When would you *not* use deep learning?](challenges/challenge-02-classical-vs-deep.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 3.

## Deliverable

A notebook that takes two photos of the same scene from different viewpoints, detects and describes keypoints in each, matches them, filters the matches, and draws the surviving correspondences as lines between the images.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
