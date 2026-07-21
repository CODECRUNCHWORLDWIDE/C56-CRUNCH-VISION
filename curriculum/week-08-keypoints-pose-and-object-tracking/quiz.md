# Week 8 — Quiz

Ten questions. Answer key below.

**1. Keypoint estimation is usually done by:**

- A. Predicting a heatmap per keypoint and taking its peak
- B. Drawing boxes
- C. Segmenting the background
- D. Directly regressing coordinates only

**2. A sharp heatmap peak vs. a diffuse one indicates:**

- A. Nothing
- B. Higher vs. lower confidence in the keypoint location
- C. More vs. fewer pixels
- D. A bigger object

**3. Human pose is keypoints plus:**

- A. A skeleton defining which keypoints connect
- B. A class label
- C. A mask
- D. A bounding box only

**4. Top-down pose estimation:**

- A. Works only on faces
- B. Detects all keypoints then groups them
- C. Detects each person first, then estimates keypoints in each box
- D. Ignores people

**5. Bottom-up pose estimation is preferred when:**

- A. There is exactly one person
- B. No keypoints are needed
- C. Only boxes are wanted
- D. Scenes are crowded and real-time matters (constant cost)

**6. Tracking-by-detection works by:**

- A. Tracking without any detector
- B. Detecting each frame, then associating detections to existing identities
- C. Segmenting every pixel
- D. Using only the first frame

**7. A Kalman filter helps tracking by:**

- A. Labeling classes
- B. Removing duplicates
- C. Computing IoU
- D. Predicting each track's next position (and smoothing), then correcting with detections

**8. Deep SORT reduces identity switches by adding:**

- A. More detectors
- B. An appearance embedding so identities match by look, not just position
- C. A larger IoU threshold
- D. Random IDs

**9. The signature failure mode of tracking is:**

- A. Wrong color
- B. Slow loading
- C. Blurry images
- D. Identity switches when objects cross/occlude

**10. Tracking should be evaluated with:**

- A. Identity-aware metrics like IDF1/HOTA (plus MOTA), not just per-frame detection
- B. Pixel accuracy
- C. Training loss
- D. Per-frame mAP only

---

## Answer key

1. **A. Predicting a heatmap per keypoint and taking its peak** — Heatmap regression is dense, represents confidence, and trains more stably than raw coordinates.
2. **B. Higher vs. lower confidence in the keypoint location** — Peak sharpness encodes the model's certainty about the keypoint position.
3. **A. A skeleton defining which keypoints connect** — The skeleton connects joints (shoulder–elbow–wrist) into a readable posture.
4. **C. Detects each person first, then estimates keypoints in each box** — Detect-then-pose-each is accurate for few people; cost grows with crowd size.
5. **D. Scenes are crowded and real-time matters (constant cost)** — Detecting all keypoints once then grouping scales better to many people.
6. **B. Detecting each frame, then associating detections to existing identities** — The whole problem reduces to associating per-frame detections across time.
7. **D. Predicting each track's next position (and smoothing), then correcting with detections** — Predict-then-correct motion modeling bridges gaps and stabilizes tracks.
8. **B. An appearance embedding so identities match by look, not just position** — Appearance features keep IDs stable when objects cross or occlude.
9. **D. Identity switches when objects cross/occlude** — ID swaps are the #1 tracking failure; appearance modeling mitigates them.
10. **A. Identity-aware metrics like IDF1/HOTA (plus MOTA), not just per-frame detection** — Consistent identity over time needs identity-aware metrics; per-frame mAP ignores swaps.
