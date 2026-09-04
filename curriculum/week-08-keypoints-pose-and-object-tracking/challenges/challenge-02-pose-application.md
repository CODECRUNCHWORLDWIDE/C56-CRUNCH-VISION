# Challenge 2 — A pose-based analyzer, with a fairness and privacy audit

Use pose keypoints to *judge* an action — the core of fitness, rehab, and gesture apps — and
confront its limits and its ethics honestly.

1. Pick a simple, well-defined task: count squat/pushup repetitions, classify a few yoga poses, detect
   arms-raised, or recognize a hand gesture.
2. Compute your signal from keypoint geometry (e.g. hip-knee-ankle angle for a squat; count reps by hysteresis
   thresholding the angle over time to avoid double-counting jitter). No new training — reason from the joint
   coordinates and their OKS confidences (down-weight or reject low-confidence joints).
3. **Robustness sweep.** Test across different people, camera angles, clothing, and lighting. Where does it
   fail — side views, occluded joints, loose clothing, unusual body types, wheelchairs or atypical morphology?
4. **Fairness and privacy audit.** Pose models carry dataset bias (COCO/MPII skew toward certain populations
   and viewpoints). Write who your analyzer works and fails for, and — because this processes body video —
   state the consent basis, what you store versus discard, and what you would refuse to build (e.g. covert
   monitoring).

**Deliverable:** a working pose-based analyzer with a demo, a robustness table naming who/what it fails on,
and an explicit fairness + privacy section. Reasoning geometrically from keypoints — and being honest about
bias and consent — is the graded skill.
