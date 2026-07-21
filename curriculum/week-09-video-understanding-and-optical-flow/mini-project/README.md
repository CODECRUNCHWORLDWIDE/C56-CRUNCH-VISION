# Mini-Project — Motion & Action on Video

## Brief

Add the time dimension to your vision toolkit: compute motion with optical flow and recognize actions across frames, while confronting video's real costs and comparing honestly against a single-frame baseline.

## Requirements

1. **Optical flow:** estimate dense flow on a short clip and visualize it as a color-coded motion field; briefly explain the brightness-constancy assumption and one place it breaks.
2. **Action recognition:** run a pretrained (Kinetics) video model on several clips with correct frame sampling and preprocessing; report top predictions and confidences.
3. **Baseline:** a single-frame (or frame-averaged) classifier compared to the temporal model, identifying which actions actually needed temporal context.
4. **Cost:** report the compute/latency difference and whether temporal modeling was justified for your clips.
5. **Honesty & ethics:** failure modes on real footage, plus a privacy statement (action recognition on people is surveillance-adjacent — consent, lawful use, bias).
6. **README:** reproduce steps and honest limitations.

## Stretch

- A flow-based application: motion trigger, stabilization, or frame interpolation (Challenge 1).
- The accuracy-vs-cost frontier across frame counts and resolutions (Challenge 2).

## What you're proving

You can model motion and time — the dimension that separates video from images — and you reason about its steep costs rather than reaching for the biggest model. Next week you meet the architecture reshaping all of vision: the Transformer, applied to images.
