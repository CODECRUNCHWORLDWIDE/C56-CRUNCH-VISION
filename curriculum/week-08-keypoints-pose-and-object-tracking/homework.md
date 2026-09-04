# Week 8 — Homework

Cement keypoints, filtering, assignment, and identity-aware evaluation before Week 9's move to full video understanding and optical flow. Do the Kalman and assignment work by hand before coding — the algebra must be muscle memory before a library hides it. Keep the ethics of tracking people in view throughout.

## Tasks

- Explain, with the spatial-structure and uncertainty arguments, why heatmap regression beats direct coordinate regression, and describe how soft-argmax / DSNT recovers a differentiable sub-pixel coordinate.
- Write the tracking-by-detection loop in pseudocode including the Kalman predict/update equations, gated Hungarian association, and track birth/death.
- Work one Kalman predict-update cycle by hand for a 1-D position filter, and state how the Kalman gain changes as measurement noise R grows or shrinks.
- Given a small 3x3 cost matrix, find the optimal assignment by the Hungarian method and show a greedy row-wise assignment is worse.
- For a specified clip failure (an ID switch and a missed detection), state which of MOTA / IDF1 / HOTA-DetA-AssA best exposes each, and why.
- Read the ByteTrack paper and explain, in a paragraph, why associating low-score detections in a second pass reduces fragmentation, and when it would instead add false tracks.

## Definition of done

A committed project that runs a pretrained pose estimator (skeletons, one decoded + visualized heatmap with the argmax-vs-soft-argmax gap, and one OKS) and implements a multi-object tracker with a Kalman filter, gated Hungarian association, track birth/death, an appearance embedding, and a ByteTrack low-score pass on a short clip; keeps consistent IDs; reports MOTA/IDF1/HOTA with identity-switch counts before/after the appearance cue and maps each failure to its metric; and includes an explicit consent/lawful-use and refusal statement with every hyperparameter documented.

Submit by committing your work to your course repo under `week-08/`.
