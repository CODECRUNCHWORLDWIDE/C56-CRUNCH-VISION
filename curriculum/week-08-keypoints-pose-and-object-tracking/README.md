# Week 8 — Keypoints, pose & object tracking

> **Goal:** by Sunday you can explain keypoint/pose estimation as heatmap regression, run a pretrained pose model, and build a multi-object tracker that keeps consistent IDs across frames using detection + IoU/appearance association.

So far, one image at a time. This week adds two capabilities that bridge to video: **keypoint and pose estimation** — locating specific points like joints, facial landmarks, or object corners — and **object tracking** — following the same object across frames with a stable identity. Pose powers fitness apps, motion capture, and sign-language recognition; tracking powers sports analytics, traffic monitoring, and any system that must count or follow. You'll run a pose model and build a tracker with the classic detect-then-associate recipe.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** keypoint estimation as per-keypoint heatmap regression, and pose as a set of connected keypoints.
- **Run** a pretrained pose estimator and interpret its keypoints and skeleton.
- **Distinguish** top-down vs. bottom-up pose approaches and their trade-offs.
- **Build** a multi-object tracker that associates detections across frames and maintains IDs.
- **Evaluate** tracking with identity-aware metrics and reason about when tracking fails.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Keypoints and pose estimation](lecture-notes/01-keypoints-and-pose.md)
2. [Lecture 2 — Object tracking across frames](lecture-notes/02-tracking-fundamentals.md)
3. [Lecture 3 — Evaluating tracking & motion honestly](lecture-notes/03-evaluating-tracking.md)

**Exercises**

1. [Exercise 1 — Run a pose estimator and read heatmaps](exercises/exercise-01-run-pose.md)
2. [Exercise 2 — Build an IoU-based tracker](exercises/exercise-02-simple-tracker.md)
3. [Exercise 3 — Find and reduce identity switches](exercises/exercise-03-diagnose-id-swaps.md)

**Challenges**

1. [Challenge 1 — Count objects and measure motion](challenges/challenge-01-count-and-measure.md)
2. [Challenge 2 — Build a pose-based analyzer](challenges/challenge-02-pose-application.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 9.

## Deliverable

A notebook or script that (a) runs a pretrained pose estimator on images/video and draws skeletons, and (b) implements a simple IoU-based multi-object tracker on a short clip that keeps consistent IDs across frames, with a note on where identities get swapped.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
