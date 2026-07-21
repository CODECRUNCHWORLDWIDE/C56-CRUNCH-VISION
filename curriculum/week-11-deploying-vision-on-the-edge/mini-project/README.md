# Mini-Project — Optimize and Deploy a Vision Model to the Edge

## Brief

Turn a notebook model into a deployable one: shrink it, export it, and benchmark it honestly for a real edge scenario — proving you can optimize accuracy under hard hardware constraints.

## Requirements

1. **Baseline:** a trained model (your transfer model from Week 5, or a pretrained one) with recorded float32 size, latency (median + tail), and held-out accuracy.
2. **Optimize:** apply at least quantization (INT8); optionally switch to an efficient architecture, prune, or distill — measuring the accuracy/size/latency delta at each step.
3. **Export:** to ONNX (or TorchScript/TFLite), run it in the target runtime, and **verify output parity and preprocessing parity** with the original (include a deliberate-mismatch demo of the failure).
4. **Benchmark:** the accuracy–latency–size trade-off across your optimization steps, measured on the shipped artifact, on the target hardware. Plot the trade-off.
5. **Recommendation:** a stated edge scenario with constraints, and a justified pick from your configurations.
6. **README:** reproduce steps, the real benchmark numbers, and honest limitations (including production monitoring and distribution-shift risks).

## Stretch

- Hit a hard latency budget (Challenge 1) and document the path.
- Quantization-aware training if post-training quantization dropped accuracy too far.

## What you're proving

You can make a vision model *real* — small enough, fast enough, and portable enough to run where it's actually needed, with honest numbers for the version you'd ship. This is the last piece before the capstone, where you take a full computer-vision system from problem to deployed, trustworthy product.
