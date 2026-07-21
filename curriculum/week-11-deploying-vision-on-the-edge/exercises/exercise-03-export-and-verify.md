# Exercise 3 — Export to ONNX and verify parity

**Goal:** get a model out of PyTorch and prove it still works.

## Tasks

1. Export a trained model to ONNX with `torch.onnx.export`, naming inputs/outputs.
2. Load it in ONNX Runtime and run inference on the same test images.
3. **Verify parity:** confirm the ONNX outputs match the PyTorch outputs to a small numerical tolerance. Investigate any mismatch (unsupported op, shape issue).
4. Explicitly reconstruct the preprocessing in the ONNX pipeline and confirm it matches training preprocessing exactly — then show what happens to accuracy if you deliberately mismatch it (wrong normalization or channel order).

## Deliverable

A notebook that exports to ONNX, confirms output parity with PyTorch, and demonstrates the preprocessing-mismatch failure. The deliberate-mismatch demo makes the #1 deployment bug unforgettable.
