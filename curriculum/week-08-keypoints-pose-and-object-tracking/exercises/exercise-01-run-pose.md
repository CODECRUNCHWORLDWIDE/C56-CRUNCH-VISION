# Exercise 1 — Run a pose estimator and read heatmaps

**Goal:** produce and interpret keypoints and a skeleton.

## Tasks

1. Load a pretrained pose model (torchvision `keypointrcnn_resnet50_fpn`, or a MediaPipe/other pose library) and run it on images of people.
2. Draw the detected keypoints and connect them into a skeleton using the standard connection set.
3. If the model exposes heatmaps, visualize the heatmap for one keypoint (e.g. right wrist) and confirm its peak is where the joint is; note the peak's sharpness as confidence.
4. Test on a hard case — partial occlusion, unusual pose — and observe which keypoints get low confidence or wrong positions.

## Deliverable

A notebook drawing skeletons on your images, one visualized keypoint heatmap, and a note on the model's confidence/failures on the hard case. Understanding the heatmap makes pose non-magical.
