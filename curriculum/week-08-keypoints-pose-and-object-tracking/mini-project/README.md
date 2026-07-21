# Mini-Project — Pose + Multi-Object Tracking on Video

## Brief

Bring images to life across time: estimate pose and track multiple objects with stable identities on a short video, proving you understand keypoint heatmaps, the detect-then-associate paradigm, and identity-aware evaluation.

## Requirements

1. **Pose:** run a pretrained pose estimator on images or video and draw skeletons; visualize at least one keypoint heatmap and explain it.
2. **Tracking:** implement a multi-object tracker on a short clip — detect per frame, associate by IoU (optimal assignment if you handle many objects), manage track birth/death, and keep consistent IDs.
3. **Robustness:** add an appearance cue (color histogram or backbone embedding) and show it reduces identity switches at a crossing/occlusion.
4. **Evaluation:** count identity switches (by eye or against a few labeled frames), and describe the failure modes (swaps, fragmentation, drift).
5. **Ethics:** a short, explicit statement — tracking people is privacy-sensitive; state consent/lawful-use and what you would refuse to build.
6. **README:** reproduce steps and honest limitations.

## Stretch

- Build a real analytic on top: line-crossing count or speed estimation (Challenge 1).
- A pose-based rep counter or gesture recognizer (Challenge 2).

## What you're proving

You can locate fine-grained keypoints and maintain object identity over time — the bridge from static images to video. Next week you go fully temporal: understanding actions in video and computing motion with optical flow.
