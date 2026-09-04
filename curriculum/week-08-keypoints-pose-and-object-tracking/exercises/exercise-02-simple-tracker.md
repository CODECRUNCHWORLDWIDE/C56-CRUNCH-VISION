# Exercise 2 — Build a Kalman + Hungarian tracker (SORT from scratch)

**Goal:** implement the detect-predict-associate loop with a real motion filter and optimal
assignment.

## Tasks

1. On a short clip (a few seconds, split into frames), run your object detector per frame to get boxes.
2. **Kalman filter.** Implement a per-track constant-velocity Kalman filter with state
   `[u, v, s, r, u_dot, v_dot, s_dot]` (centre, scale, aspect, velocities). Code the `predict`
   (`x <- Fx`, `P <- F P F^T + Q`) and `update` (`K = P H^T S^{-1}`, `x <- x + K y`, `P <- (I - KH)P`) steps.
3. **Association.** Each frame, build the cost matrix `cost[i][j] = 1 - IoU(predicted_track_i, detection_j)`,
   **gate** matches with `IoU < 0.3` to infinity, and solve with `scipy.optimize.linear_sum_assignment`
   (Hungarian). Update matched tracks, birth unmatched detections (tentative -> confirmed after N_init),
   kill tracks unmatched for `A_max` frames.
4. Assign and draw a persistent ID and colour per track across the clip.
5. Play back the annotated frames; confirm IDs stay consistent for steadily-moving objects. Then **ablate**:
   replace the Kalman prediction with "last box" (no motion model) and report where tracking degrades.

## Deliverable

An annotated clip where each object keeps a consistent ID, your tracker code (Kalman + Hungarian), and a
short note on the Kalman-vs-no-motion ablation. This is SORT, hand-built.
