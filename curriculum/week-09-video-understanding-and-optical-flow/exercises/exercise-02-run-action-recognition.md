# Exercise 2 — Run an action recognizer

**Goal:** classify what's *happening* in a clip, not just what's in a frame.

## Tasks

1. Load a pretrained video classification model (torchvision video models like R(2+1)D or a Kinetics-pretrained network).
2. Prepare a few short clips (yours or public), sampling the required number of frames and preprocessing exactly as the model expects.
3. Run the model and report the top predicted actions with confidences for each clip.
4. Test a clip whose action is ambiguous from a single frame (e.g. sitting down vs. standing up) and confirm the temporal model handles it.

## Deliverable

A notebook reporting action predictions on your clips, including the single-frame-ambiguous case, plus the frame-sampling and preprocessing you used. Note any confidently-wrong predictions.
