# Exercise 1 — Run a pose estimator, decode heatmaps, and score with OKS

**Goal:** produce and interpret keypoints and a skeleton, and connect the heatmap to a coordinate
and a confidence quantitatively.

## Tasks

1. Load a pretrained pose model (torchvision `keypointrcnn_resnet50_fpn`, or MediaPipe / MMPose) and run it
   on several images of people. Draw the 17 COCO keypoints and connect them into a skeleton with the standard
   connection set.
2. **Heatmap decoding.** If the model exposes per-keypoint heatmaps, pick one joint (e.g. right wrist) and
   decode its coordinate three ways: (a) bare `argmax`; (b) argmax with the quarter-pixel offset toward the
   second-highest neighbour; (c) **soft-argmax** (normalize with softmax, take the expected grid coordinate).
   Report the three coordinates and the pixel gap between (a) and (c) — that gap is the quantization error.
3. Visualize the chosen keypoint's heatmap as an overlay. Confirm the peak sits on the joint and note that a
   sharp peak means high confidence, a diffuse blob low confidence.
4. **OKS by hand.** For one image with ground-truth (or hand-labelled) keypoints, compute the OKS between the
   prediction and truth using the COCO per-keypoint constants; interpret the number against the OKS-AP
   thresholds.
5. Test a hard case — partial occlusion, unusual pose, crowd — and note which keypoints get low confidence or
   wrong positions.

## Deliverable

A notebook drawing skeletons, one visualized heatmap, the three decoded coordinates with the argmax-vs-
soft-argmax gap in pixels, a computed OKS for one image, and a note on the model's failures on the hard case.
Understanding the heatmap-to-coordinate path makes pose non-magical.
