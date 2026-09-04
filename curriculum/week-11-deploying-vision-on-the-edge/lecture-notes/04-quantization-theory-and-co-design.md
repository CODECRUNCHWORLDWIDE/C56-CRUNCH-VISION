# Lecture 4 — Quantization theory, QAT with the straight-through estimator, & hardware-aware co-design

Lecture 2 gave the affine-quantization algebra and named QAT; this lecture proves the piece that makes
QAT possible — differentiating through a non-differentiable rounding function — and then broadens to the
frontier: mixed precision, learned quantization, and **hardware-aware neural architecture search**, where the
model and the deployment target are designed together.

## Why post-training quantization is not enough

PTQ perturbs a network the training never anticipated: every weight and activation is snapped to a grid, and
the errors, though individually small (`~s/2`), *accumulate through depth* and can knock a sensitive layer off
its operating point. Depthwise layers are notoriously fragile — their per-channel weight distributions vary
wildly, so a per-tensor scale clips some channels to near-zero. When PTQ's drop exceeds your budget, you train
*through* the quantizer. But the quantizer `q(x) = s·(round(x/s) − z... )` has derivative **zero almost
everywhere** (it is piecewise constant) and undefined at the steps — gradients cannot flow. This is the
central obstacle QAT must overcome.

## The straight-through estimator (STE)

The fix is the **straight-through estimator** (Bengio, Léonard & Courville, 2013, "Estimating or propagating
gradients through stochastic neurons"). In the *forward* pass we apply the real, discretizing quantizer — the
network sees true INT8 behavior (a **fake-quant** node: quantize then dequantize, so downstream math stays
float but carries quantization error). In the *backward* pass we pretend the rounding was the identity:

    forward:   x_q = s · round(clip(x/s) − z ... )      (non-differentiable)
    backward:  dL/dx = dL/dx_q · 1{ x within clip range }   (STE: treat round' = 1)

That is, `∂round(u)/∂u := 1` inside the representable range and `0` where the input is clipped (so clipped
values get no gradient — a "dead zone"). The STE is a **biased** gradient estimator — it is not the true
gradient of the (a.e. flat) function — yet it works remarkably well: the network learns weights whose
quantized values sit in low-loss regions, effectively pre-compensating for the rounding it will suffer at
inference (Jacob et al., 2018; Krishnamoorthi, 2018). QAT typically recovers most or all of the PTQ gap, at
the cost of a fine-tuning run.

## Learned and mixed-precision quantization

- **Learned Step Size Quantization (LSQ)** (Esser et al., 2020, ICLR) makes the scale `s` itself a trained
  parameter, with an STE-style gradient `∂x_q/∂s`, so the network *learns* the optimal grid rather than fixing
  it by calibration. It is a strong, simple QAT baseline, especially at low bit-widths.
- **PACT** (Choi et al., 2018) learns the activation clipping bound, directly optimizing the rounding/clipping
  trade-off.
- **Mixed precision.** Not all layers are equally sensitive. Keep the fragile ones (often the first conv and
  final classifier) in INT8 or FP16 while pushing robust middle layers to INT4. **HAWQ** (Dong et al., 2019,
  ICCV) uses the **Hessian** — a layer with large second-order curvature is sensitive, so it deserves more
  bits — to assign per-layer precision principledly rather than by trial and error.
- **Extreme quantization.** Binary/ternary nets (XNOR-Net, Rastegari et al., 2016) replace multiplies with
  bit-ops for huge speed/energy wins at real accuracy cost — viable for tightly constrained, tolerant tasks.

## Batch-norm folding, a QAT gotcha

At inference, BatchNorm is an affine transform that should be **folded** into the preceding conv's weights and
bias (one fused op — again a memory-bound win). During QAT you must simulate quantization of the *folded*
weights, and handle BN's running statistics carefully, or training-time and inference-time numerics diverge.
Frameworks provide fused Conv-BN modules for exactly this; getting it wrong is a common QAT accuracy bug.

## Hardware-aware co-design: search the model *for* the chip

The deepest idea in efficient deployment is that architecture and hardware should be designed **together**.
Early neural architecture search (NAS) optimized accuracy alone using thousands of GPU-days. Modern
**hardware-aware NAS** puts *measured on-device latency* into the objective:

- **MnasNet** (Tan et al., 2019, CVPR) searches with a multi-objective reward `accuracy · [latency/target]^w`,
  measuring latency on the actual phone — because FLOPs mispredict latency (Lecture 1's roofline: a low-FLOP
  depthwise layer can be slower than a high-FLOP dense one).
- **ProxylessNAS** (Cai, Zhu & Han, 2019, ICLR) makes the search differentiable and uses a **latency lookup
  table** as a differentiable cost, searching directly on the target task/hardware.
- **Once-for-All** (Cai et al., 2020, ICLR) trains one super-network from which specialized sub-networks can be
  extracted for many devices *without retraining* — decoupling training cost from the number of deployment
  targets.
- **MCUNet** (Lin et al., 2020, NeurIPS) co-designs the network (TinyNAS) and an inference engine (TinyEngine)
  to fit ImageNet-scale vision into a **microcontroller's 320 KB SRAM** — the extreme of co-design (Lecture 5).

The unifying principle: optimize the *end-to-end deployed metric on the real target*, not a proxy. FLOPs,
parameter counts, and dev-GPU latency are all proxies that can mislead.

**Takeaway:** QAT beats PTQ because it trains *through* the quantizer using the straight-through estimator
(Bengio et al., 2013) — forward pass applies real fake-quant rounding, backward pass treats rounding as the
identity inside the clip range — a biased but effective gradient that lets weights pre-compensate for rounding.
Learned scales (LSQ), Hessian-guided mixed precision (HAWQ), and careful BN folding push this further; and the
frontier is hardware-aware co-design (MnasNet, ProxylessNAS, Once-for-All, MCUNet), which searches the
architecture against *measured on-device latency* because every proxy — FLOPs included — can lie.
