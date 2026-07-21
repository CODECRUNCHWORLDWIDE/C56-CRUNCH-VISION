# Mini-Project — The Capstone

## Brief

Design, build, train, evaluate, and deploy one computer-vision system end to end on a problem you choose. This is the course, proven.

## Requirements

1. **Framing:** a problem that matters to you, a task (classification/detection/segmentation), a dataset you're allowed to use (licenses, privacy, consent), a clear metric, and a documented baseline.
2. **Model:** an appropriate architecture (you have CNNs, transfer learning, detectors, segmenters, ViTs), trained with the training toolbox and transfer learning applied deliberately.
3. **Result:** beat the baseline on **held-out images** — or honestly analyze why you didn't. Show the win is robust to randomness.
4. **Honest evaluation:** the task's right metric vs. baseline, a visual error analysis on mistakes, a per-subgroup / dataset-bias check, and named failure modes.
5. **Deployment:** the model exported and served behind a `/predict` API with **verified preprocessing parity**; if edge-targeted, optimized and benchmarked per Week 11. Reproducible setup (pinned deps, entry points, saved artifacts).
6. **Communication:** a README that leads with results and limitations, and a complete model card addressing privacy, consent, and bias.

## Definition of done

Someone can clone your repo, follow the README, reproduce your headline result, send the API a real image and get a sensible prediction, and read an honest account of what the system does, where it fails, and for whom.

## What you're proving

Everything. Twelve weeks ago an image was a mystery grid of numbers; now you can take a computer-vision problem from nothing to a trustworthy, deployed, documented system — and defend every technical and ethical choice in it. That's a computer-vision engineer. Where next? [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/) to ship and monitor at scale, [C53 Crunch Nets](../C53-CRUNCH-NETS/) for deeper architecture theory, and [C5 Crunch AI & Data Science](../C5-CRUNCH-AI-DATA-SCIENCE/) to widen the foundation.
