# Mini-Project — Motion & Action on Video: Flow, Recognition, and the Honesty Check

## Brief

Add the time dimension to your vision toolkit: compute motion with optical flow, recognize actions
across frames with a pretrained video model, and — crucially — confront video's real costs and compare
honestly against a single-frame baseline. This is the deliverable that proves you can read *time*, not
just appearance, and that you reason about video's steep cost instead of reaching for the biggest model.

## Requirements

1. **Optical flow.** Estimate dense flow on a short clip and visualize it as a color-coded motion field
   (hue = direction, value = magnitude, with a legend). Briefly explain the brightness-constancy
   assumption and identify **one** concrete place it breaks in your clip (occlusion, specular highlight,
   fast motion, or lighting change), naming which assumption failed.
2. **Action recognition.** Run a pretrained (Kinetics) video model on several clips with **correct frame
   sampling and preprocessing** and the right `(B,C,T,H,W)` tensor order. Report top-5 predictions and
   confidences. Include at least one clip whose action is ambiguous from a single frame and confirm the
   temporal model resolves it.
3. **Single-frame baseline (the truth serum).** Build a single-frame (or frame-averaged) classifier on
   the same clips and compare it to the temporal model **per action**. Identify which actions actually
   needed temporal context and which were appearance-defined.
4. **Cost.** Report the compute/latency and peak-memory difference between the temporal model and the
   baseline, and state a clear verdict: *was* temporal modeling justified for your clips?
5. **Evaluation hygiene.** State explicitly that any accuracy comparison splits **by video**, not by
   frame, and why frame-splitting would leak.
6. **Honesty & ethics.** Report failure modes on real footage (camera motion, blur, viewpoint), and a
   substantive privacy statement: action recognition on people is surveillance-adjacent — address
   consent, lawful use, dataset bias (Kinetics' geographic/demographic skew), and the mirror test.
7. **README.** Reproduce steps, exact model + preprocessing, and honest limitations.

## Stretch

- **Classical vs. learned flow:** add RAFT and compare against Farneback via warping residual (Ex. 4).
- **A flow-based application:** motion trigger, stabilization, or frame interpolation (Challenge 1).
- **The accuracy-vs-cost frontier** across frame counts and resolutions with a Pareto plot (Challenge 2).

## What you are proving

You can model motion and time — the dimension that separates video from images — with the flow math
underneath it, and you treat video's cost as an engineering variable to be measured, not ignored. You
also treat behavior recognition on people with the heightened privacy care it demands. Next week you
meet the architecture reshaping all of vision: the Transformer, applied to images.
