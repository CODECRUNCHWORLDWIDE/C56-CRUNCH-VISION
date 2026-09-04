# Exercise 3 — Export to ONNX, verify output parity, and break preprocessing on purpose

**Goal:** get a model out of PyTorch, prove it still works, and make the #1 deployment bug unforgettable.

## Tasks

1. Export a trained model to ONNX with `torch.onnx.export` (name inputs/outputs, set an opset). Load it in
   ONNX Runtime and run inference on the same fixed test batch.
2. **Verify output parity:** assert the ONNX logits match the PyTorch logits to a tolerance
   (`np.allclose(..., atol=1e-4)`). If they do not, diagnose it — unsupported op, NCHW/NHWC layout, or a
   trace-vs-script control-flow issue — and fix it.
3. **Reconstruct preprocessing** in the deployment pipeline (resize interpolation, center-crop, channel order,
   per-channel mean/std, [0,1] scaling) and confirm it matches training preprocessing exactly on held-out
   accuracy.
4. **Deliberate mismatch:** now break preprocessing three ways — swap RGB->BGR, drop the normalization, and
   change the resize interpolation — re-running held-out accuracy after each. Record how far accuracy falls
   while the export "passes" every structural check.

## Deliverable

A notebook that exports to ONNX, confirms output parity with PyTorch, matches preprocessing to reach baseline
accuracy, and then demonstrates the accuracy collapse from each deliberate preprocessing mismatch — with a
short reflection on why offline tests miss this class of bug.
