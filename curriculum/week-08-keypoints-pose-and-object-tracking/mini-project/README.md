# Mini-Project — Pose + Multi-Object Tracking on Video, Evaluated Honestly

## Brief

Bring images to life across time: estimate pose and track multiple objects with stable identities on a short
video, proving you understand keypoint heatmaps, the Kalman-filter + optimal-assignment paradigm, and
identity-aware evaluation — and that you can reason about the privacy footing of tracking people.

## Requirements

1. **Pose.** Run a pretrained pose estimator on images or video and draw skeletons. Visualize at least one
   keypoint heatmap and decode its coordinate with both argmax and soft-argmax, reporting the sub-pixel gap.
   Compute an OKS for one labelled frame and interpret it.
2. **Tracking.** Implement a multi-object tracker on a short clip: a constant-velocity **Kalman filter** per
   track (predict / update), **Hungarian** association on a gated IoU cost, and track birth/death. Keep
   consistent IDs and draw them.
3. **Robustness.** Add an **appearance embedding** (colour histogram or backbone crop features) and
   Mahalanobis gating, and show it reduces identity switches at a crossing/occlusion. Then add **ByteTrack's
   low-score second pass** and report whether it further recovers occluded tracks.
4. **Evaluation.** On a labelled snippet, compute **MOTA, IDF1, and (if available) HOTA with DetA/AssA**.
   Count identity switches, and map each observed failure (swap, fragmentation, drift, phantom) to the metric
   that exposes it.
5. **Ethics.** A short, explicit statement: tracking people is privacy-sensitive and, for biometric/real-time
   public use, legally regulated (GDPR lawful basis; EU AI Act). State the consent/lawful-use basis of your
   clip, what you retain versus discard, and what you would refuse to build.
6. **README.** Reproduce steps, hyperparameters (Q, R, gates, lambda), and honest limitations.

## Stretch

- Build a real analytic on top: line-crossing unique-count or calibrated speed estimation (Challenge 1).
- A pose-based rep counter or gesture recognizer with a fairness audit (Challenge 2).
- Swap in a transformer tracker or a JDE/FairMOT joint model and compare metrics and speed (Challenge 3).

## Definition of done

A committed project that runs a pretrained pose estimator (skeletons + one decoded/visualized heatmap + one
OKS) and implements a multi-object tracker (Kalman predict/update, Hungarian association with gating, track
birth/death, an appearance embedding, and a ByteTrack low-score pass) on a short clip; keeps consistent IDs;
reports MOTA/IDF1/HOTA with identity-switch counts before/after the appearance cue; maps failures to metrics;
and includes an explicit consent/lawful-use and refusal statement. Every hyperparameter (Q, R, gates, lambda,
A_max) is documented and justified.

## What you are proving

You can locate fine-grained keypoints via heatmap regression and maintain object identity over time with a
principled filter + assignment tracker, evaluate it with the right metrics, and situate people-tracking in its
legal and ethical frame. Next week you go fully temporal: understanding actions in video and computing motion
with optical flow.
