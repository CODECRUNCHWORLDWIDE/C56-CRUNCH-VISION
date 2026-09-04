# Challenge 3 — TinyML memory wall or compiler autotuning (open, research-flavored)

An open challenge — pick ONE track and investigate rigorously. Negative or partial results, well
analyzed, earn full credit; the graded skill is experimental rigor.

**Track A — the TinyML memory wall.** Take an efficient classifier and try to make it fit an aggressive **peak
activation memory** budget (say 256-512 KB), the true binding constraint on microcontrollers (Lin et al.,
2020, MCUNet). Profile the *per-layer* activation memory and find where the peak is (usually an early
high-resolution layer). Attack it: reduce input resolution, add early downsampling, restructure to a
patch-based/streaming schedule, quantize activations to INT8. Plot peak activation memory vs. accuracy across
your attempts and characterize the frontier. Explain why parameter count and FLOPs are the *wrong* metric here.

**Track B — compiler autotuning.** Take one exported model and run it through at least two runtimes/compilers
(e.g. plain ONNX Runtime vs. ONNX Runtime with graph optimizations, or a TVM-tuned build), on the same
hardware. Measure the latency spread from software alone — same model, same numerics — and attribute it to
fusion, layout, and scheduling (Lecture 5). If using TVM, run its autotuner and report the tuning-time vs.
speedup curve.

**Deliverable:** a short report with figures, a precise statement of what you observed, and an honest account
of what your experiment does and does not establish. For Track A, the activation-memory frontier and where the
peak lives; for Track B, the software-only latency spread and its attribution. Argue carefully rather than
overclaim.
