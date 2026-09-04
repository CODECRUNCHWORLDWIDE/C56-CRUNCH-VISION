# Week 9 — Homework

Cement motion and temporal modeling before Vision Transformers. These take a few focused hours and set up Week 10's move to attention-based vision. Do the flow and cost derivations by hand before coding — the math must be muscle memory before a library hides it, and the baseline/leakage discipline must be instinct before you evaluate anything on real footage.

## Tasks

- Re-derive, on paper without notes, the optical flow constraint from brightness constancy, and explain the aperture problem in your own words as a rank deficiency of the structure tensor `M`.
- Write the Horn-Schunck energy and describe its two terms and the role of `α`; then list the rungs of the temporal-modeling ladder (frame aggregation → 3D → two-stream → (2+1)D → spacetime attention) and the cost each rung adds.
- Explain, quantitatively, why splitting a video dataset by frame leaks and how splitting by whole video fixes it; give a one-line estimate of how much a frame-split can inflate accuracy.
- Read the RAFT abstract and the TimeSformer abstract; in a short paragraph each, state how RAFT replaces hand-designed smoothness and how divided space-time attention cuts the quadratic cost.
- Extend your mini-project to add the single-frame baseline and report per-action accuracy plus measured latency, then write one sentence per action stating whether temporal modeling was justified.
- Compute (by hand) the self-attention cost ratio of joint vs. divided space-time attention for `N=196`, `T=8`, and state which axis you would scale first and why.

## Definition of done

A committed notebook that: estimates and color-visualizes dense optical flow on a clip with a named brightness-constancy failure; runs a Kinetics-pretrained video model with correct frame sampling and `(B,C,T,H,W)` order, reporting top-5 actions including a single-frame-ambiguous case; compares a single-frame baseline per action and states whether temporal modeling earned its measured compute cost; evaluates with a by-video split; and includes real-footage failure modes plus a substantive consent/bias/mirror-test privacy statement — with a README that reproduces every step.

Submit by committing your work to your course repo under `week-09/`.
