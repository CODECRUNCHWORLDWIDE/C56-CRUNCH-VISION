# Lecture 4 — The mathematics of association: Kalman filtering, gating & optimal assignment

Lecture 2 used the Kalman filter and the Hungarian algorithm as black boxes. This lecture opens
both. The payoff is that "predict then correct" and "optimal matching" become equations you can derive,
extend, and debug — and you will understand *why* SORT and Deep SORT are principled, not lucky.

## The linear-Gaussian state-space model

Model each track's hidden state `x_t` (e.g. box centre, size, and velocities) evolving linearly with
Gaussian noise, observed linearly with Gaussian noise:

    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)      (motion / process model)
    z_t = H x_t + v_t,       v_t ~ N(0, R)      (measurement model)

`F` is the constant-velocity transition, `H` selects the observable part (the box, not the velocity), `Q`
is process noise (how much you distrust the constant-velocity assumption), `R` is measurement noise (how
much you distrust the detector). The **Kalman filter** (Kalman, 1960) computes the *exact* posterior
`p(x_t | z_{1:t})`, which for this linear-Gaussian model stays Gaussian, so it is fully described by a mean
`x_hat` and covariance `P`.

## Predict

Push the previous posterior through the motion model. Because the model is linear and the noise Gaussian,
the predicted state is Gaussian with

    x_hat^- = F x_hat_{t-1},
    P^-      = F P_{t-1} F^T + Q.

Read `P^- = F P F^T + Q`: uncertainty is propagated by the dynamics and then *inflated* by `Q`. The future
is less certain than the present — exactly the intuition, now an equation.

## Update (the Bayesian correction)

A measurement `z_t` arrives. Form the **innovation** (measurement minus prediction) and its covariance:

    y = z_t - H x_hat^-,     S = H P^- H^T + R.

The **Kalman gain** is `K = P^- H^T S^{-1}`, and the posterior is

    x_hat = x_hat^- + K y,    P = (I - K H) P^-.

`K` is the crux. If measurement noise `R` is small (a trusted detector), `K` is large and the update leans on
`z_t`; if `R` is large, `K` is small and the update keeps the prediction. This is precisely inverse-variance
weighting — the optimal linear combination of two Gaussian estimates. One can prove `x_hat` is both the MMSE
(minimum-mean-squared-error) estimator and, because everything is Gaussian, the MAP estimate; the filter is
the *optimal* estimator for this model class, not a heuristic (see Thrun, Burgard & Fox, *Probabilistic
Robotics*, 2005, Ch. 3, for the full derivation).

## Worked micro-example (1-D)

Position-only, `F = H = 1`. Prior `x_hat = 10`, `P = 4`. Process noise `Q = 1` -> predict `x_hat^- = 10`,
`P^- = 5`. Detector measures `z = 13` with `R = 5`. Then `S = 5 + 5 = 10`, `K = 5/10 = 0.5`,
`x_hat = 10 + 0.5*(13 - 10) = 11.5`, `P = (1 - 0.5)*5 = 2.5`. The estimate moved halfway to the measurement
(equal variances) and its uncertainty *dropped* from 5 to 2.5. Equal trust -> split the difference; the
correction shrinks covariance.

## Gating with the Mahalanobis distance

The innovation covariance `S` gives a natural, scale-aware distance between a track prediction and a
detection: the squared **Mahalanobis distance** `d^2 = y^T S^{-1} y`. Under the Gaussian model `d^2` is
chi-squared distributed with degrees of freedom equal to the measurement dimension, so a threshold from the
chi-squared table (Deep SORT uses the 0.95 quantile, `d^2 < 9.49` for 4-D box measurements) rejects
implausible matches *before* assignment. This is the principled generalization of an IoU floor: it accounts
for *direction* and *magnitude* of uncertainty, not just overlap.

## The assignment problem and the Hungarian algorithm

With gated costs `c_ij` (cost of matching track `i` to detection `j`), association is the **linear
assignment problem**: choose a permutation minimizing total cost. As an integer program,

    minimize  sum_ij c_ij x_ij   s.t.  sum_j x_ij = 1,  sum_i x_ij = 1,  x_ij in {0,1}.

The constraint matrix is **totally unimodular**, so the LP relaxation (`x_ij >= 0`) has integral vertices —
you can drop the integrality constraint and still get a 0/1 solution. The **Hungarian algorithm** (Kuhn,
1955, building on Konig's theorem; Munkres, 1957) solves it in O(n^3) by manipulating the dual: it maintains
row/column potentials (dual variables) and augments along equality-tight edges until a perfect matching of
zero reduced cost exists. Complementary slackness guarantees that matching is optimal. Practically you call
`scipy.optimize.linear_sum_assignment`, but knowing it is exact LP-dual optimization — not greedy — is why
you trust it in crowds where greedy nearest-neighbour strands the globally better pairing.

## Deep SORT's fused cost, assembled

Deep SORT's cost matrix combines the two machines above:

    c_ij = lambda * d_motion(i, j) + (1 - lambda) * d_appearance(i, j),

with `d_motion` the Mahalanobis distance from the Kalman prediction (also used for gating) and
`d_appearance` the smallest cosine distance between detection `j`'s embedding and the gallery of track `i`'s
recent embeddings. A **matching cascade** prioritizes recently-seen tracks so a long-occluded track does not
hijack a detection from an actively-tracked one. Every piece is now derived: the filter, the gate, the
solver, and their fusion.

## Common pitfalls

- **Miscalibrated Q and R.** Too-small `R` makes the filter chase every jittery detection; too-small `Q`
  makes it ignore real manoeuvres and drift. These two numbers *are* the tracker's personality — tune them.
- **Skipping gating.** Feeding un-gated costs to Hungarian lets it minimize total cost with an absurd
  long-range match. Gate first, assign second.
- **Assuming linearity holds.** Constant-velocity fails for sharp turns and camera motion; that is where the
  extended/unscented Kalman filter, or the learned motion of transformer trackers (Lecture 5), earn their keep.

**Takeaway:** the Kalman filter is the exact posterior for a linear-Gaussian state — predict inflates
covariance by `Q`, update shrinks it via the inverse-variance Kalman gain `K = P^- H^T S^{-1}`. Its
innovation covariance yields a chi-squared Mahalanobis gate. Association is a totally-unimodular assignment
LP the Hungarian algorithm solves optimally in O(n^3) via its dual. Deep SORT is exactly these parts fused —
principled end to end.
