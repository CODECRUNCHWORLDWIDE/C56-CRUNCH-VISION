# Lecture 5 — Compilers, runtimes, edge accelerators, & TinyML

A quantized, pruned model is only as fast as the software stack that runs it and the silicon underneath.
This lecture opens the black box below "the runtime": how deep-learning **compilers** turn a graph into fast
kernels, what the zoo of **edge accelerators** actually does, and how **TinyML** pushes vision onto
microcontrollers — with the safety and privacy framing that on-device inference deserves.

## From graph to kernels: the deep-learning compiler

Between your ONNX file and the metal sits a compiler whose job is to schedule computation for a specific
target. The performance levers, in roughly increasing sophistication:

- **Operator fusion.** Fold chains like conv → BN → ReLU (and elementwise adds) into a single kernel so
  intermediate activations never touch DRAM. Because edge inference is frequently memory-bound (Lecture 1),
  fusion is often the single biggest wall-clock win — it raises arithmetic intensity by cutting bytes moved.
- **Layout selection.** Choose NCHW vs. NHWC (and tiled/blocked layouts) to match the hardware's preferred
  memory access and vector width. The wrong layout can halve throughput.
- **Memory planning.** Statically allocate and *reuse* activation buffers across non-overlapping tensor
  lifetimes, minimizing peak SRAM — the binding constraint on microcontrollers.
- **Tiling and vectorization.** Block loops to fit cache/SRAM and map inner loops onto SIMD/tensor units.
- **Autotuning.** **Apache TVM** (Chen et al., 2018, OSDI, "TVM: An automated end-to-end optimizing compiler
  for deep learning") searches over schedules — tile sizes, unroll factors, orderings — and uses a learned
  cost model to find fast kernels per operator per target, rather than relying on hand-written libraries.
  **XLA** (TensorFlow/JAX) and **TensorRT** occupy similar roles with more fixed strategies.

The takeaway for a deployer: the same ONNX graph can run 2-10x apart depending on the compiler and its
settings. Benchmark the *compiled* artifact (Lecture 3), and know that "the model" and "the kernels" are
separable optimization targets.

## The edge accelerator zoo

General CPUs run anything but inefficiently. Purpose-built silicon trades flexibility for FLOP/watt:

- **Mobile GPUs** (Adreno, Mali) — good parallel throughput; reached via Vulkan/OpenCL delegates.
- **DSPs** (e.g. Qualcomm Hexagon) — very efficient for INT8 vision, often the best perf/watt on a phone.
- **NPUs / TPUs** — dedicated neural accelerators with systolic MAC arrays (Google Edge TPU, Apple Neural
  Engine). They are typically **INT8-first** — which is *why* quantization is not optional but the price of
  admission to the fast path. An FP32 model may fall back to the slow CPU entirely.
- **Jetson-class edge GPUs** (NVIDIA) — CUDA + TensorRT for higher-power robots/drones.
- **Microcontrollers** (ARM Cortex-M) — kilobytes to a few hundred KB of SRAM, no OS, milliwatts of power.

A vital point: an accelerator only helps if your ops map to it. An unsupported operator triggers a **fallback**
to CPU, and a graph that ping-pongs between NPU and CPU can be *slower* than staying on CPU because of transfer
overhead. Deployment engineering is partly keeping the whole graph on the accelerator's happy path.

## TinyML: vision in a few hundred kilobytes

**TinyML** runs inference on microcontrollers — always-on, battery-or-harvested-power, no cloud. The binding
constraint is **peak activation memory (SRAM)**, not FLOPs or even model size, because a single
high-resolution early-layer feature map can exceed total SRAM. **MCUNet** (Lin et al., 2020, NeurIPS)
co-designs TinyNAS (a network whose activation memory fits) with TinyEngine (a code-generating runtime doing
in-place depthwise convs and patch-based execution) to run ImageNet-grade classification on a device with
~320 KB SRAM and ~1 MB flash. **TensorFlow Lite for Microcontrollers** provides the standard interpreter for
this regime. Techniques that matter here: INT8 everywhere, patch-based/streaming execution to bound peak
memory, and operator sets pruned to what the MCU supports.

## On-device inference is a privacy and safety property, not just a speed trick

Running vision at the edge changes the system's ethics, and a graduate treatment must say so plainly:

- **Privacy.** On-device inference means image data need never leave the device — a strong privacy guarantee
  for cameras in homes, hospitals, or public spaces. This is a *design* choice worth making deliberately: keep
  raw imagery local, transmit only derived, minimal signals, and be explicit about what (if anything) is sent
  to the cloud. On-device is often the more privacy-preserving architecture *by construction*.
- **Safety.** Edge vision frequently drives real-world action — a drone avoiding obstacles, a car's perception
  stack, a medical screening device. Latency and its *tail* are safety-critical: a p99 spike that misses a
  frame can mean a missed obstacle. Quantization changes the model's error surface, so **re-validate accuracy
  and failure modes on the shipped, quantized artifact**, not the FP32 original, before anything acts on its
  output. Test on the real distribution and near-distribution edge cases.
- **Robustness and monitoring.** Field conditions drift from training data (lighting, weather, sensor aging).
  Because the edge device may be offline, plan how you will detect distribution shift and degradation — logging
  confidence, sampling for audit, staged rollout — a direct handoff to [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/).

**Takeaway:** a deployed model's speed is set as much by the compiler and silicon as by the network. DL
compilers (TVM's autotuning, XLA, TensorRT) win chiefly through fusion, layout, memory planning, and tiling —
often memory-bound wins — so benchmark the *compiled* artifact and keep the whole graph on the accelerator's
INT8-first happy path (which is *why* quantization is the price of admission to NPUs). TinyML (MCUNet, TFLite
Micro) fits vision into a few hundred KB of SRAM by treating peak activation memory as the binding constraint.
And on-device inference is a privacy guarantee and a safety responsibility: keep raw imagery local, and
re-validate the *quantized, shipped* model's accuracy and tail latency before it drives any real-world action.
