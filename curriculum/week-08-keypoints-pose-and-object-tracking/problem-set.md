# Week 8 — Graduate Problem Set: Estimation, Filtering, Assignment, and Evaluation

Ten problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution
sketches follow — attempt each fully first. Notation: a track's Kalman state has predicted mean `x^-`,
covariance `P^-`, measurement matrix `H`, measurement noise `R`; the innovation is `y = z - H x^-` with
covariance `S = H P^- H^T + R` and gain `K = P^- H^T S^{-1}`.

**P1 (heatmap target).** A keypoint is at continuous location `(u, v) = (10.4, 6.7)` on a heatmap. Write the
value of the Gaussian target `T(x, y) = exp(-[(x-u)^2 + (y-v)^2]/(2 sigma^2))` with `sigma = 1` at the four
integer pixels surrounding it. Which pixel is the argmax, and what is the argmax quantization error in each
axis relative to `(u, v)`?

**P2 (soft-argmax removes bias).** For a 1-D heatmap over pixels `{0,1,2,3,4}` with softmax-normalized
weights `p = [0.02, 0.13, 0.35, 0.35, 0.15]`, compute the argmax coordinate and the soft-argmax (expected)
coordinate. Explain in one sentence why soft-argmax is differentiable and argmax is not.

**P3 (OKS).** A person has scale `s = 100` (sqrt area in px). Two keypoints have errors `d_1 = 5` px
(k_1 = 0.025, an eye) and `d_2 = 20` px (k_2 = 0.107, a hip), both visible. Compute the per-keypoint
similarities and the OKS. Which keypoint dominates the score loss, and why does the per-keypoint constant
matter?

**P4 (Kalman gain, 1-D).** Position-only filter, `F = H = 1`. Prior `x = 20`, `P = 9`, process noise
`Q = 1`, measurement `z = 26` with `R = 3`. Compute `x^-`, `P^-`, `S`, `K`, the posterior `x`, and posterior
`P`. Verify the posterior covariance is smaller than both the predicted covariance and `R`.

**P5 (gain limits).** From the 1-D update, prove that as `R -> 0` the posterior mean `x -> z` and as
`R -> infinity` it `-> x^-`. Interpret both limits in words (trusted vs. distrusted detector).

**P6 (Mahalanobis gating).** A 2-D innovation is `y = [3, 4]^T` with `S = diag(4, 4)`. Compute the squared
Mahalanobis distance `d^2 = y^T S^{-1} y`. If the chi-squared 0.95 threshold for 2 dof is 5.99, is this match
accepted or rejected? Contrast with the Euclidean distance `||y||`.

**P7 (assignment optimality).** Give the 2x2 cost matrix `[[1, 4], [3, 2]]`. Enumerate both possible
assignments, give their total costs, and state the optimal one. Then explain why a greedy "assign each row to
its cheapest column" rule can be suboptimal, using this or a slightly modified matrix.

**P8 (MOTA vs. IDF1 sensitivity).** A 100-frame clip has one ground-truth object present every frame. Tracker
A produces a correct box every frame but switches its ID once at frame 50. Tracker B misses the object in 10
frames but never switches ID. With `GT = 100`, compute MOTA for each (FP = 0 for both; A: IDSW = 1, FN = 0;
B: IDSW = 0, FN = 10). Argue qualitatively which tracker IDF1 prefers and why the two metrics disagree.

**P9 (HOTA geometric mean).** Tracker X has `DetA = 0.9`, `AssA = 0.4`; tracker Y has `DetA = 0.6`,
`AssA = 0.6`. Compute HOTA for each. Which does HOTA prefer, and what does the geometric mean (versus an
arithmetic mean) express about balancing detection and association?

**P10 (open — ByteTrack analysis).** ByteTrack keeps low-score detections and associates them in a second
pass by motion only, with no appearance model, yet rivals appearance-based trackers on MOT17/MOT20. Argue
*why* this works: what property of low-score boxes makes them recoverable, why motion (not appearance) is
sufficient in the second pass, and in what scenario the low-score pass would instead *inject* false tracks.
Relate your answer to DetA vs. AssA. (Open-ended; reason carefully.)

---

## Solution sketches

**S1.** Nearest pixels `(10,6),(11,6),(10,7),(11,7)`. Distances-squared from `(10.4,6.7)`:
`(0.16+0.49)=0.65`, `(0.36+0.49)=0.85`, `(0.16+0.09)=0.25`, `(0.36+0.09)=0.45`; targets `exp(-d^2/2)` =
`0.72, 0.65, 0.88, 0.80`. Argmax is `(10,7)`; quantization error is `(0.4, 0.3)` px — up to ~0.5 px per axis.
**S2.** Argmax = pixel 2 or 3 (tie at 0.35; conventionally the first, 2). Soft-argmax =
`0*.02+1*.13+2*.35+3*.35+4*.15 = 0.13+0.70+1.05+0.60 = 2.48`. Soft-argmax is a weighted sum (smooth in the
weights) so its gradient exists; argmax is a discontinuous selection with zero/undefined gradient.
**S3.** `sim_1 = exp(-25/(2*100^2*0.025^2)) = exp(-25/12.5) = exp(-2.0) = 0.135`;
`sim_2 = exp(-400/(2*100^2*0.107^2)) = exp(-400/229) = exp(-1.747) = 0.174`. OKS = mean = ~0.155. The eye's
small `k` makes even a 5 px error costly, showing the constant sets per-joint difficulty.
**S4.** `x^- = 20`, `P^- = 9+1 = 10`; `S = 10+3 = 13`; `K = 10/13 = 0.769`;
`x = 20 + 0.769*(26-20) = 24.6`; `P = (1-0.769)*10 = 2.31`. Posterior 2.31 < both 10 and 3.
**S5.** `x = x^- + [P^-/(P^-+R)](z - x^-)`. As `R->0`, weight ->1, `x->z`. As `R->inf`, weight ->0, `x->x^-`.
Trusted detector: follow the measurement; useless detector: keep the prediction.
**S6.** `S^{-1} = diag(0.25, 0.25)`; `d^2 = 0.25*9 + 0.25*16 = 6.25 > 5.99` -> reject. Euclidean `||y|| = 5`
would look moderate; Mahalanobis normalizes by covariance and, being calibrated to chi-squared, rejects it.
**S7.** Assignment (r0->c0, r1->c1) costs `1+2 = 3`; (r0->c1, r1->c0) costs `4+3 = 7`. Optimal = 3. Greedy by
row: r0 picks c0 (cost 1), r1 then forced to c1 (2) — here greedy happens to match; modify to
`[[1,2],[1,9]]`: greedy gives r0->c0 (1), r1->c1 (9) = 10, but optimal is r0->c1 (2), r1->c0 (1) = 3. Greedy
is not globally optimal.
**S8.** A: `MOTA = 1 - (0+0+1)/100 = 0.99`. B: `MOTA = 1 - (0+10+0)/100 = 0.90`. So MOTA prefers A. But B
never breaks identity, so IDF1 (trajectory-level) prefers B — the single ID switch fragments A's identity
into two, which IDF1 penalizes heavily. The metrics disagree because MOTA is detection-dominated and nearly
identity-blind.
**S9.** X: `sqrt(0.9*0.4) = sqrt(0.36) = 0.60`. Y: `sqrt(0.6*0.6) = 0.60`. Tie here; but push X to
`AssA=0.3`: `sqrt(0.27)=0.52 < 0.60`. The geometric mean punishes imbalance — a tracker cannot buy HOTA by
being excellent at one sub-task and poor at the other, unlike an arithmetic mean.
**S10.** Low-score boxes are frequently *real* objects degraded by occlusion/blur, so their location is still
informative; a leftover track's Kalman prediction overlaps the true (low-score) box, so motion alone
disambiguates — appearance is unreliable exactly when the crop is occluded/blurred anyway. The pass injects
false tracks when low-score boxes are genuine background false positives that happen to overlap a predicted
track (crowded, cluttered scenes); this trades a small DetA/FP cost for a large AssA/fragmentation gain,
which is why it wins on identity-heavy benchmarks.
