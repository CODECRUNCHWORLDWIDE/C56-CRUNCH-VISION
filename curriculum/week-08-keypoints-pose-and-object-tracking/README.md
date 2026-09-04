# Week 8 — Keypoints, pose & object tracking

> **Goal:** by Sunday you can (1) explain keypoint estimation as per-keypoint heatmap regression, derive why a rendered-Gaussian target beats direct coordinate regression, and reason about soft-argmax / DSNT, integral regression, and the OKS metric; (2) distinguish top-down and bottom-up multi-person pipelines and state the grouping mathematics (part-affinity-field line integrals, associative embeddings); (3) derive the Kalman filter as the exact posterior recursion for a linear-Gaussian state, cast frame-to-frame association as an optimal assignment problem solved by the Hungarian algorithm, and build SORT / Deep SORT from those parts; and (4) evaluate a tracker with the identity-aware metrics MOTA, IDF1, and HOTA — knowing precisely what each measures, where each misleads, and what the privacy law around tracking people requires of you.

Through Week 7 you have worked one image at a time. This week adds the two capabilities that bridge to **video**: **keypoint / pose estimation** — locating named points such as joints, facial landmarks, or object corners — and **multi-object tracking (MOT)** — following the *same* object across frames with a stable identity. Pose powers motion capture, rehabilitation, sign-language recognition, and 6-DoF object pose for robotics; tracking powers sports analytics, traffic monitoring, and any system that must count, measure speed, or reason about behaviour over time.

The undergraduate version of this week hand-waves 'run a pose model, then match boxes by overlap.' We will not. You will see *why* heatmap regression is the dominant estimation paradigm (Tompson et al., 2014) rather than the direct coordinate regression of DeepPose (Toshev & Szegedy, 2014), and how soft-argmax / DSNT (Nibali et al., 2018) recovers differentiable, sub-pixel coordinates from a heatmap. You will **derive the Kalman filter** as the closed-form Bayesian posterior for a linear-Gaussian state model (Kalman, 1960), so 'predict then correct' becomes an equation you can write, not a slogan. You will cast association as the **linear assignment problem** and solve it optimally with the Hungarian algorithm (Kuhn, 1955), understanding it as an LP whose integral optimum the algorithm finds in O(n^3). And you will meet the metric family — CLEAR-MOT / MOTA (Bernardin & Stiefelhagen, 2008), IDF1 (Ristani et al., 2016), and HOTA (Luiten et al., 2021) — as precise definitions, not acronyms, so you know exactly which failure each rewards or hides.

Two advanced lectures push to the research frontier: the exact mathematics of filtering, gating, and optimal assignment; and the modern trackers that fold detection, embedding, and association into one network — JDE / FairMOT, ByteTrack's low-score association trick, and the transformer trackers (TrackFormer, MOTR) that treat tracks as persistent queries. Throughout, one non-negotiable thread: **tracking people is privacy-sensitive and often legally regulated.** We treat consent, lawful basis, and refusal as first-class engineering constraints, not footnotes.

## Learning objectives

By the end of this week, you will be able to:

- **Derive** keypoint estimation as per-keypoint heatmap regression against a rendered-Gaussian target, and explain quantitatively why it trains more stably than direct coordinate regression.
- **Recover** differentiable sub-pixel coordinates from heatmaps via soft-argmax / DSNT and integral regression, and state the argmax quantization error they remove.
- **Contrast** top-down and bottom-up multi-person pipelines, and explain the two canonical grouping mechanisms — part-affinity-field line integrals and associative embeddings.
- **Prove** that the Kalman filter is the exact posterior mean and covariance for a linear-Gaussian state-space model, and compute one predict-update cycle by hand.
- **Formulate** frame-to-frame association as a linear assignment problem, solve it with the Hungarian algorithm, and gate matches by Mahalanobis distance and appearance cosine distance.
- **Define** MOTA, IDF1, and HOTA from their bipartite-matching constructions, and diagnose which tracker failure (ID switch, fragmentation, drift, false track) each metric exposes or masks.
- **Build** a multi-object tracker (SORT then Deep SORT) end to end, and demonstrate that an appearance embedding reduces identity switches at a crossing.
- **Assess** the privacy and legal footing of a people-tracking system — consent, lawful basis, data minimisation — and state, in writing, what you would refuse to build.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CAP 4410` — estimate keypoints and human pose, and track multiple objects across the frames of a video. |
| Industry | Keep object identities stable through a crossing, and report the identity switches before and after the change that was supposed to fix them. |
| Beyond the bar | proves the Kalman filter is the exact posterior for a linear-Gaussian model and works one predict-update cycle by hand before any library is used — `lecture-notes/04-association-mathematics.md` |

## Prerequisites

- Week 6 (object detection) and Week 7 (segmentation / encoder-decoder heatmap heads), committed and working.
- Linear algebra: matrix multiplication, inverse, positive-definite / covariance matrices, eigenvalues.
- Probability: multivariate Gaussians, conditional distributions, Bayes' rule, expectation and covariance.
- Basic optimization / graph ideas: bipartite matching and the notion of an optimal assignment.

## This week

**Lectures**

1. [Lecture 1 — Keypoints & pose: heatmap regression, sub-pixel decoding, and OKS](lecture-notes/01-keypoints-and-pose.md)
2. [Lecture 2 — Tracking-by-detection: the Kalman filter and data association](lecture-notes/02-tracking-fundamentals.md)
3. [Lecture 3 — Evaluating tracking honestly: MOTA, IDF1, and HOTA derived](lecture-notes/03-evaluating-tracking.md)
4. [Lecture 4 — The mathematics of association: Kalman filtering, gating & optimal assignment](lecture-notes/04-association-mathematics.md)
5. [Lecture 5 — Frontiers: joint detection-embedding, ByteTrack, and transformer trackers](lecture-notes/05-frontiers-joint-and-transformer-tracking.md)

**Exercises**

1. [Exercise 1 — Run a pose estimator, decode heatmaps, and score with OKS](exercises/exercise-01-run-pose.md)
2. [Exercise 2 — Build a Kalman + Hungarian tracker (SORT from scratch)](exercises/exercise-02-simple-tracker.md)
3. [Exercise 3 — Reduce identity switches with appearance (toward Deep SORT)](exercises/exercise-03-diagnose-id-swaps.md)
4. [Exercise 4 — Compute MOTA / IDF1 and add ByteTrack's low-score pass](exercises/exercise-04-metrics-and-bytetrack.md)

**Challenges**

1. [Challenge 1 — Count uniques and measure motion, with an error budget](challenges/challenge-01-count-and-measure.md)
2. [Challenge 2 — A pose-based analyzer, with a fairness and privacy audit](challenges/challenge-02-pose-application.md)
3. [Challenge 3 — Tracker bake-off: SORT vs. Deep SORT vs. ByteTrack (open)](challenges/challenge-03-tracker-bakeoff-open.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 9.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook or script that (a) runs a pretrained pose estimator on images/video, draws skeletons, and visualizes at least one keypoint heatmap with its OKS-style confidence; and (b) implements a multi-object tracker on a short clip — Kalman prediction, Hungarian association, track birth/death, and an appearance embedding — that keeps consistent IDs, reports identity switches at a crossing before/after the appearance cue, and carries an explicit consent / lawful-use statement.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
