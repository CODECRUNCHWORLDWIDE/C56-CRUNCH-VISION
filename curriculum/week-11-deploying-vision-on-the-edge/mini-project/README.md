# Mini-Project — Optimize and Deploy a Vision Model to the Edge, Honestly

## Brief

Turn a notebook model into a deployable one: shrink it, export it, and benchmark it honestly for a real edge
scenario with a **hard constraint** — proving you can maximize accuracy subject to hardware limits and report
the numbers for the version you would actually ship.

## Requirements

1. **Baseline.** A trained model (your Week-5 transfer model, or a pretrained one) with recorded FP32 on-disk
   size, latency (median *and* p95/p99, after warmup, batch 1), peak memory, and held-out accuracy — all on
   the target hardware. State the target explicitly.
2. **Roofline read.** Profile FLOPs and identify which layers are compute- vs. memory-bound, so your
   optimizations target the right thing (do not cut FLOPs on a bandwidth-bound layer expecting speed).
3. **Optimize, measuring at every step.** Apply at least INT8 quantization (PTQ; use QAT if PTQ drops too
   far). Optionally also switch to an efficient architecture, structured-prune, or distill. Record the
   accuracy/size/latency delta after each step, and note per-channel vs. per-tensor and the calibration method
   you chose and why.
4. **Export + parity.** Export to a portable runtime (ONNX/TorchScript/TFLite). **Verify output parity**
   (allclose to PyTorch) and **preprocessing parity** end-to-end, and include a **deliberate-mismatch demo**
   (wrong channel order or normalization) showing the silent accuracy collapse.
5. **Benchmark the shipped artifact.** Measure the exported, quantized model on the target: median+tail
   latency, peak memory, size, and held-out accuracy. Plot the **accuracy-latency (and accuracy-size)
   frontier** across FP32 -> optimized steps.
6. **Recommendation.** State an edge scenario with a *hard constraint* (e.g. "<=33 ms p99 on device X") and
   defend a Pareto-optimal pick from your configurations with a short decision rationale.
7. **README.** Reproduce steps, the real benchmark numbers, and honest limitations — including production
   monitoring and distribution-shift risks, and (if the scenario is safety/privacy-sensitive) how on-device
   inference and quantized re-validation address them.

## Stretch

- Hit a hard **tail-latency** budget (Challenge 1) and document the path through configuration space.
- Do **QAT with your own straight-through fake-quant** (Exercise 4) if PTQ dropped accuracy too far, and show
  the recovered gap.
- Push toward a TinyML **peak-activation-memory** budget (Challenge 3, Track A) and plot the memory frontier.

## What you are proving

You can make a vision model *real* — small enough, fast enough, portable enough, and *honestly measured*
enough to run where it is actually needed, with the numbers reported for the artifact you would ship. This is
the last piece before the capstone, where you take a full computer-vision system from problem to deployed,
trustworthy product.
