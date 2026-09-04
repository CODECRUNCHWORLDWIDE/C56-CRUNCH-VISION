# Exercise 4 — Compute MOTA / IDF1 and add ByteTrack's low-score pass

**Goal:** measure your tracker with identity-aware metrics and implement the single highest-leverage
association upgrade.

## Part A — metrics

1. Hand-label (or obtain) ground-truth boxes with identities for a short clip. Run your tracker and, using a
   metrics library (`motmetrics` / `TrackEval`), compute **MOTA**, **IDF1**, and — if available — **HOTA with
   its DetA / AssA decomposition**.
2. Deliberately degrade the tracker two ways: (i) lower the detector score threshold to add false positives;
   (ii) shorten `A_max` so tracks die during occlusion (more fragmentation / ID switches). Report how MOTA,
   IDF1, and HOTA each respond, and confirm that ID switches crush IDF1/AssA while barely moving MOTA.

## Part B — ByteTrack low-score association

1. Split detections into high-score (> 0.5) and low-score (0.1-0.5). Associate tracks to high-score
   detections first; then associate the *remaining unmatched tracks* to low-score detections using motion
   (IoU with the Kalman prediction). Discard leftover low-score boxes.
2. Re-run the metrics from Part A. Report the change in IDF1/AssA and fragmentation. Explain, with a specific
   occlusion frame, why keeping low-score boxes recovered a track.

## Deliverable

A metrics table (MOTA / IDF1 / HOTA-DetA-AssA) for baseline, degraded, and ByteTrack variants, plus a
paragraph explaining which metric exposed which failure and why the low-score pass helped (Zhang et al.,
2022).
