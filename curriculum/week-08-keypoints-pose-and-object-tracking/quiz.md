# Week 8 — Quiz

Fifteen questions spanning heatmap estimation and decoding, OKS, the Kalman filter and gating, optimal assignment, the MOTA/IDF1/HOTA metric family, and modern trackers. Attempt each before the answer key.

**1. Why is heatmap regression preferred over direct coordinate regression for keypoints?**

- A. It uses fewer parameters than a coordinate head
- B. It requires no ground-truth labels
- C. It avoids using convolutions entirely
- D. It preserves translation-equivariant spatial structure, represents uncertainty, and trains more stably

<details>
<summary>Answer</summary>

**D. It preserves translation-equivariant spatial structure, represents uncertainty, and trains more stably** — A rendered-Gaussian heatmap keeps the spatial prior convolutions exploit, encodes confidence in peak sharpness, and trains more stably than regressing a global coordinate vector (Tompson et al., 2014).

</details>

**2. The training target for a keypoint heatmap is usually:**

- A. the bounding box of the person
- B. a one-hot delta at the ground-truth pixel
- C. a 2-D Gaussian rendered at the ground-truth location
- D. the full segmentation mask of the person

<details>
<summary>Answer</summary>

**C. a 2-D Gaussian rendered at the ground-truth location** — A Gaussian target injects tolerance and smooth gradients near the joint; a hard delta gives almost no learning signal away from the exact pixel.

</details>

**3. Soft-argmax / DSNT decoding computes a keypoint coordinate as:**

- A. the expected value of the coordinate grid under the softmax-normalized heatmap
- B. the centroid of all pixels above 0.5
- C. the bounding-box centre
- D. the pixel with the maximum heatmap value

<details>
<summary>Answer</summary>

**A. the expected value of the coordinate grid under the softmax-normalized heatmap** — Soft-argmax takes the expectation of the grid coordinate under the normalized heatmap, giving a differentiable, sub-pixel coordinate free of argmax quantization (Nibali et al., 2018).

</details>

**4. OKS (Object Keypoint Similarity) differs from raw pixel error because it:**

- A. is identical to bounding-box IoU
- B. ignores keypoint visibility
- C. normalizes error by object scale and a per-keypoint difficulty constant
- D. measures only the hardest keypoint

<details>
<summary>Answer</summary>

**C. normalizes error by object scale and a per-keypoint difficulty constant** — OKS divides squared error by object scale and a per-keypoint falloff k_i, making it comparable across image sizes and joints; it plays the role IoU plays for boxes.

</details>

**5. In the tracking-by-detection loop, the step that carries all the difficulty is:**

- A. running the detector
- B. data association (matching detections to existing tracks)
- C. reading the video file
- D. drawing the boxes

<details>
<summary>Answer</summary>

**B. data association (matching detections to existing tracks)** — Detection is Week 6; the memory that makes tracking is deciding which new detection continues which existing identity — the association step.

</details>

**6. In the Kalman predict step, the covariance update P^- = F P F^T + Q reflects that:**

- A. uncertainty shrinks over time
- B. uncertainty is propagated by the dynamics and then inflated by process noise
- C. the measurement is trusted completely
- D. the state is unobservable

<details>
<summary>Answer</summary>

**B. uncertainty is propagated by the dynamics and then inflated by process noise** — Prediction pushes covariance through the motion model F and adds Q, so predicted uncertainty grows — the future is less certain than the present.

</details>

**7. The Kalman gain K = P^- H^T S^{-1} behaves so that when measurement noise R is large:**

- A. K is large and the update follows the detection closely
- B. K equals the identity matrix
- C. the filter diverges
- D. K is small and the update keeps the prediction

<details>
<summary>Answer</summary>

**D. K is small and the update keeps the prediction** — Large R means a distrusted detector, so S is large, K is small, and the posterior leans on the motion prediction — inverse-variance weighting.

</details>

**8. Deep SORT reduces identity switches during crossings chiefly by adding:**

- A. a larger IoU threshold
- B. a learned appearance embedding matched by cosine distance
- C. more detectors
- D. random re-initialization of IDs

<details>
<summary>Answer</summary>

**B. a learned appearance embedding matched by cosine distance** — An appearance (re-ID) embedding lets identities match by look, which survives occlusion and disambiguates overlapping same-class boxes that IoU alone confuses (Wojke et al., 2017).

</details>

**9. Gating association matches by the Mahalanobis distance d^2 = y^T S^{-1} y is principled because:**

- A. it ignores the innovation covariance
- B. it is faster than computing IoU
- C. d^2 is always zero for a correct match
- D. under the Gaussian model d^2 is chi-squared distributed, giving a calibrated rejection threshold

<details>
<summary>Answer</summary>

**D. under the Gaussian model d^2 is chi-squared distributed, giving a calibrated rejection threshold** — The squared Mahalanobis distance of the innovation is chi-squared with df = measurement dimension, so a table quantile (e.g. 9.49 for 4-D at 0.95) rejects implausible matches with calibrated confidence.

</details>

**10. The Hungarian algorithm solves frame-to-frame assignment optimally because the assignment LP:**

- A. has a totally unimodular constraint matrix, so its LP relaxation has integral optima
- B. requires enumerating all permutations
- C. is solved greedily per detection
- D. is non-convex and solved by gradient descent

<details>
<summary>Answer</summary>

**A. has a totally unimodular constraint matrix, so its LP relaxation has integral optima** — The assignment constraint matrix is totally unimodular; the LP relaxation therefore has 0/1 vertex solutions, and the Hungarian algorithm finds one in O(n^3) via LP duality (Kuhn, 1955).

</details>

**11. MOTA is often criticized because it:**

- A. is detection-dominated and can score high despite many identity errors
- B. cannot be computed on real data
- C. requires appearance embeddings
- D. ignores false positives

<details>
<summary>Answer</summary>

**A. is detection-dominated and can score high despite many identity errors** — Because FP + FN typically dwarf IDSW, MOTA is dominated by detection quality and can look good while identities are frequently swapped — hence IDF1 and HOTA.

</details>

**12. IDF1 differs from MOTA principally by:**

- A. using per-frame greedy matching only
- B. measuring pixel accuracy
- C. ignoring identity entirely
- D. matching at the trajectory (identity) level via a global bipartite matching

<details>
<summary>Answer</summary>

**D. matching at the trajectory (identity) level via a global bipartite matching** — IDF1 solves one global identity-to-identity matching maximizing correctly-identified frames, so it rewards keeping one predicted ID on one object for its whole life (Ristani et al., 2016).

</details>

**13. HOTA is defined (at threshold alpha) as:**

- A. the same quantity as MOTA
- B. the arithmetic mean of precision and recall
- C. the geometric mean sqrt(DetA * AssA) of detection and association accuracy
- D. 1 minus the identity-switch rate

<details>
<summary>Answer</summary>

**C. the geometric mean sqrt(DetA * AssA) of detection and association accuracy** — HOTA takes the geometric mean of detection accuracy and association accuracy and integrates over thresholds, so a tracker must be good at both and cannot hide a weak half (Luiten et al., 2021).

</details>

**14. ByteTrack improves tracking without any appearance model by:**

- A. discarding all low-confidence detections earlier
- B. training a larger re-ID network
- C. increasing the IoU threshold to 0.9
- D. associating remaining tracks to low-score detections in a second pass using motion

<details>
<summary>Answer</summary>

**D. associating remaining tracks to low-score detections in a second pass using motion** — Low-score boxes are often occluded real objects; a second association pass matches leftover tracks to them by motion, recovering occluded objects and cutting fragmentation (Zhang et al., 2022).

</details>

**15. Transformer trackers such as TrackFormer and MOTR perform association by:**

- A. running the Hungarian algorithm at every inference frame
- B. carrying track queries forward so attention implicitly re-localizes each object
- C. computing optical flow between frames
- D. using a hand-tuned Kalman filter only

<details>
<summary>Answer</summary>

**B. carrying track queries forward so attention implicitly re-localizes each object** — Track queries from frame t are reused as queries in frame t+1 and attend to the new features to regress the same object, making association implicit and end-to-end (Meinhardt et al., 2022; Zeng et al., 2022).

</details>

---
