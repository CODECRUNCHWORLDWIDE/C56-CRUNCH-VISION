# Week 9 — Video understanding & optical flow

> **Goal:** by Sunday you can (1) derive the optical-flow constraint from brightness constancy, state the aperture problem precisely, and solve for flow with Lucas-Kanade (via the structure tensor) and with a variational Horn-Schunck energy, knowing exactly when each breaks; (2) explain how modern learned flow (PWC-Net, RAFT) replaces hand-designed smoothness with a correlation volume and a learned recurrent update, and why it wins on large motion and occlusion; (3) place every video architecture — frame aggregation, 3D CNNs, (2+1)D, two-stream, and spacetime Transformers — on a rung of a cost/accuracy ladder and account for its FLOPs and memory; and (4) run a Kinetics-pretrained model on clips, compare it against a single-frame baseline as an honesty check, and reason about the surveillance-ethics weight of action recognition on people.

Video is images plus **time**, and time is not a free extra axis — it is where most of the information *and* most of the cost live. A single frame cannot distinguish 'sitting down' from 'standing up', 'opening' from 'closing', 'picking up' from 'putting down': the label is defined by *change*, and change only exists across frames. This week builds the two pillars that let a model read that change. The first is **optical flow** — the dense per-pixel motion field — which we derive from the brightness-constancy assumption, expose as fundamentally underdetermined (the aperture problem: one equation, two unknowns), and then close with extra assumptions that are either *local* (Lucas & Kanade 1981, via the structure tensor), *global* (Horn & Schunck 1981, a variational energy with Euler-Lagrange equations), or *learned* (RAFT, Teed & Deng 2020). The second is **temporal modeling** — the architectures that reason over frames — arranged as a ladder from a cheap frame-aggregation baseline up through 3D CNNs, factorized (2+1)D convolutions (Tran et al. 2018), two-stream fusion (Simonyan & Zisserman 2014), and spacetime self-attention (Bertasius et al. 2021).

The usual treatment hand-waves 'flow is arrows' and 'video models are 3D CNNs.' We will not. You will see *why* flow along an edge is unknowable from a small window (the rank of the structure tensor), *why* a variational smoothness prior fills in the ambiguous directions, and *why* a learned all-pairs correlation volume beats both on real footage. You will count the FLOPs that make 3D convolution expensive and see exactly what (2+1)D buys back. And because action recognition on people is surveillance technology, we treat consent, lawful purpose, dataset bias, and honest baseline-checking as first-class engineering — not an afterthought. No result in this week is asserted; each is derived, measured, or cited.

## Learning objectives

By the end of this week, you will be able to:

- **Derive** the optical-flow constraint equation from brightness constancy via a first-order Taylor expansion, and state precisely why it is underdetermined (the aperture problem).
- **Solve** for flow two ways — Lucas-Kanade through the 2x2 structure tensor (and its eigenvalue conditioning), and Horn-Schunck as a variational energy whose Euler-Lagrange equations you can write down.
- **Explain** how learned flow (PWC-Net's feature-pyramid warping, RAFT's all-pairs correlation volume and recurrent GRU updates) replaces hand-designed priors, and why it handles large motion and occlusion.
- **Classify** video architectures on a cost/accuracy ladder — frame aggregation, 3D CNN, (2+1)D, two-stream, spacetime Transformer — and account for the FLOP and memory cost each rung adds.
- **Factorize** spacetime self-attention (joint vs. divided space-time attention) and state the quadratic-in-tokens cost that motivates the split.
- **Run** a Kinetics-pretrained action recognizer with correct frame sampling and preprocessing, and interpret top-k predictions and confident failures on real clips.
- **Benchmark** a single-frame baseline against a temporal model as the truth-serum honesty check, and quantify whether temporal modeling earned its compute.
- **Assess** the privacy, consent, and dataset-bias stakes of behavior recognition on people, and refuse surveillance you would not accept aimed at yourself.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 4670` — estimate motion between frames with optical flow, and recognize activity in video. |
| Industry | Show whether a temporal model earned the compute it costs, by putting it against a single-frame baseline on the same clips. |
| Beyond the bar | sets the classical flow solvers against a learned one and asks for the accuracy and the cost of each, not a preference — `exercises/exercise-04-classical-vs-learned-flow.md` |

## Prerequisites

- The previous week's mini-project (keypoints / pose / tracking), committed and working.
- Image gradients and convolution (Week 1-2): Sobel, the structure tensor, corner detection.
- CNNs and transfer learning (Weeks 3-5): pretraining, fine-tuning, group-aware train/test splits.
- Basic multivariable calculus and linear algebra: Taylor expansion, least squares, eigenvalues of a 2x2 matrix.

## This week

**Lectures**

1. [Lecture 1 — Optical flow: the motion field](lecture-notes/01-optical-flow.md)
2. [Lecture 2 — Architectures for video understanding](lecture-notes/02-video-architectures.md)
3. [Lecture 3 — Video in practice: cost, data & honesty](lecture-notes/03-video-in-practice.md)
4. [Lecture 4 — Deep optical flow: correlation volumes, warping, and RAFT](lecture-notes/04-deep-optical-flow-raft.md)
5. [Lecture 5 — Spacetime self-attention & self-supervised video](lecture-notes/05-spacetime-attention-and-self-supervision.md)

**Exercises**

1. [Exercise 1 — Estimate and visualize optical flow (dense + sparse)](exercises/exercise-01-optical-flow.md)
2. [Exercise 2 — Run a Kinetics-pretrained action recognizer](exercises/exercise-02-run-action-recognition.md)
3. [Exercise 3 — The single-frame baseline (the truth serum)](exercises/exercise-03-single-frame-baseline.md)
4. [Exercise 4 — Classical vs. learned flow, and warping quality](exercises/exercise-04-classical-vs-learned-flow.md)

**Challenges**

1. [Challenge 1 — Build something real with optical flow](challenges/challenge-01-flow-application.md)
2. [Challenge 2 — Map the video cost/accuracy frontier](challenges/challenge-02-video-cost-frontier.md)
3. [Challenge 3 — Temporal action localization with an ethics gate](challenges/challenge-03-temporal-localization-and-ethics.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 10.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook that: (1) estimates dense optical flow on a short clip and renders it as a color-coded motion field (hue = direction, brightness = magnitude), with a written note on one place brightness constancy breaks; (2) runs a Kinetics-pretrained video model on several clips with correct frame sampling, reporting top-k actions and confidences, including a single-frame-ambiguous case; (3) builds a single-frame baseline on the same clips and reports the accuracy gap and the compute-cost difference; and (4) states an explicit privacy/ethics note and honest failure modes on real footage.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
