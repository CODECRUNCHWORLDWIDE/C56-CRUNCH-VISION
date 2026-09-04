# Exercise 2 — Run a Kinetics-pretrained action recognizer

**Goal:** classify what is *happening* in a clip, not just what is in a frame — with correct
sampling and preprocessing.

## Tasks

1. Load a pretrained video classification model (torchvision `r2plus1d_18` or `mc3_18` with
   `KINETICS400_V1` weights, or a Kinetics-pretrained network of your choice).
2. Prepare a few short clips (yours or public). **Sample** the required number of frames (e.g. 16),
   resize/crop, and normalize with the exact mean/std the model expects. Assemble the tensor in
   `(B, C, T, H, W)` order — get the axis order right; it is the most common bug.
3. Run the model and report the **top-5** predicted actions with confidences (softmax) for each clip.
4. Test a clip whose action is **ambiguous from a single frame** (e.g. sitting down vs. standing up,
   opening vs. closing) and confirm the temporal model resolves it.
5. Deliberately break one thing (wrong frame count, wrong normalization, or shuffled frame order) and
   record how predictions degrade — evidence the temporal structure matters.

## Deliverable

A notebook reporting top-5 action predictions on your clips including the single-frame-ambiguous case,
the exact frame-sampling and preprocessing you used, and the degradation experiment. Note any
confidently-wrong predictions and hypothesize why.
