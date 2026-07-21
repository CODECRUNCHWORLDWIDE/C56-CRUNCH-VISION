# Challenge 2 — Build a pose-based analyzer

Use pose keypoints to *judge* an action — the core of fitness, rehab, and gesture apps — and confront its limits honestly.

1. Pick a simple, well-defined task: count squat/pushup repetitions, classify a few yoga poses, detect when arms are raised, or recognize a hand gesture.
2. Compute your signal from keypoint geometry (e.g. hip–knee–ankle angle for a squat; count reps by thresholding the angle over time). No new training required — reason from the joint coordinates.
3. Test across different people, camera angles, and clothing. Where does it fail — side views, occluded joints, loose clothing, unusual body types?
4. Write an honest limitations section: for whom and in what conditions does your analyzer work or fail? (Pose models themselves carry dataset bias.)

**Deliverable:** a working pose-based analyzer with a demo and a candid limitations section naming who/what it fails on. Reasoning geometrically from keypoints — and being honest about bias — is the skill.
