# Week 11 — Graduate Problem Set: Efficiency, Quantization, and the Deployment Roofline

Ten problems, easy to hard, mixing derivation, computation, and open analysis. Solution sketches are at
the end — attempt each fully before reading them. Notation: a convolution maps `C_in -> C_out` channels with a
`k x k` kernel over an `H x W` output; INT8 uses integer range `[q_min, q_max]`.

**P1 (conv cost, warm-up).** Derive the FLOP count and parameter count of a standard `k x k` convolution and of
its depthwise-separable factorization. Show the FLOP ratio is `1/C_out + 1/k²`, and evaluate it numerically for
`k=3, C_out=256`.

**P2 (arithmetic intensity).** A `1x1` convolution over an `H x W` map maps `C_in -> C_out`. Assuming FP32
weights and activations and that each output pixel reads its inputs and weights once, estimate FLOPs and bytes
moved, and hence the arithmetic intensity `I`. If the target has peak compute `P = 100 GFLOP/s` and bandwidth
`B = 25 GB/s`, is this layer compute- or memory-bound? State the ridge point `I*`.

**P3 (quantization map).** FP32 activations lie in `[-6, 6]`. Derive the scale `s` and zero-point `z` for
signed INT8 (`[-128, 127]`) symmetric quantization, and for unsigned INT8 (`[0, 255]`) asymmetric. Quantize
`x = 1.7` under each and give the dequantized value and the rounding error.

**P4 (quantization error).** For uniform quantization with bin width `s`, model rounding error as uniform on
`[-s/2, s/2]`. Derive its mean and variance. Explain why widening the clip range reduces clipping error but
increases this rounding variance, and state the calibration objective this trade defines.

**P5 (per-tensor vs per-channel).** A conv weight tensor has two output channels with value ranges `[-0.1,
0.1]` and `[-3.0, 3.0]`. Compute the per-tensor symmetric INT8 scale and the two per-channel scales. Quantize
the value `0.05` (in channel 1) under both schemes and compare the relative error. Explain the accuracy
implication.

**P6 (STE gradient).** Let `q(x)` be the fake-quant `s*(clip(round(x/s), qmin, qmax))` (take `z=0`). State its
true derivative `dq/dx` almost everywhere. Write the straight-through estimator's *surrogate* backward, and
argue in one sentence why the surrogate — though biased — enables useful training while the true derivative
does not.

**P7 (distillation temperature).** For logits `z` and temperature `T`, `p_i = softmax(z_i/T)`. Show that as
`T -> infinity` the softened distribution approaches uniform, and as `T -> 1` it recovers the standard softmax.
Explain why intermediate `T > 1` exposes "dark knowledge," and why the distillation loss is scaled by `T²`.

**P8 (compression accounting).** A model is 100 MB in FP32. You (a) structured-prune 40% of channels, then (b)
quantize the remainder to INT8. Estimate the final size, stating your assumptions. Why is the actual latency
speedup from (a) more reliable than from an equivalent amount of *unstructured* pruning?

**P9 (benchmark statistics).** You measure per-image latency (ms) after warmup:
`[19, 20, 20, 21, 19, 55, 20, 22, 20, 21]`. Compute the mean, median, and p90. Your budget is "30 fps"
(<=33 ms). Does the model meet it by the mean? By the tail? Which should you report for a real-time stream, and
why is the `55` not safely ignorable?

**P10 (open co-design).** FLOPs and parameter count are proxies that can mislead (Lecture 1, Lecture 4).
Construct or describe a concrete case where model A has *fewer* FLOPs than model B yet runs *slower* on a given
edge target. Explain it using arithmetic intensity and/or accelerator operator support, and argue why
hardware-aware NAS (measuring on-device latency directly, e.g. MnasNet) is the principled response. (Open-ended;
argue carefully.)

---

## Solution sketches

**S1.** Standard: FLOPs `= H·W·k²·C_in·C_out`, params `= k²·C_in·C_out`. Depthwise-separable: FLOPs
`= H·W·k²·C_in + H·W·C_in·C_out`, params `= k²·C_in + C_in·C_out`. Ratio `= 1/C_out + 1/k²`; for `k=3,
C_out=256`: `1/256 + 1/9 ~ 0.004 + 0.111 = 0.115` (~8.7x cheaper).

**S2.** FLOPs `~ H·W·C_in·C_out` (2 per MAC). Bytes `~ 4·(C_in·C_out weights + H·W·C_in inputs + H·W·C_out
outputs)`. `I = FLOPs/bytes`; for typical sizes `I` is low (few FLOP/byte) → memory-bound. Ridge point
`I* = P/B = 100/25 = 4 FLOP/byte`; if `I < 4`, memory-bound.

**S3.** Symmetric signed: `s = 6/127 ~ 0.0472`, `z = 0`; `q = round(1.7/0.0472)=36`, `x_hat = 36·0.0472 =
1.700`, error `~0.0`-ish (< s/2 ~ 0.0236). Asymmetric unsigned over `[-6,6]`: `s = 12/255 ~ 0.0471`,
`z = round(0 - (-6)/s) = 127`; `q = round(1.7/0.0471)+127 = 36+127 = 163`, `x_hat = 0.0471·(163-127)=1.696`,
error ~0.004.

**S4.** Mean `= 0`; variance `= ∫_{-s/2}^{s/2} (1/s)e² de = s²/12`. Widening the range raises `s` (more
rounding variance) but captures more values (less clipping). Calibration minimizes total error — e.g. KL
between FP32 and quantized histograms — trading the two.

**S5.** Per-tensor scale `= 3.0/127 ~ 0.0236`; channel-1 scale `= 0.1/127 ~ 0.000787`, channel-2 `~ 0.0236`.
Quantizing `0.05`: per-tensor `q = round(0.05/0.0236)=2`, `x_hat=0.0472`, rel err ~5.6%; per-channel
`q=round(0.05/0.000787)=64`, `x_hat=0.0504`, rel err ~0.8%. Per-channel is far more accurate for the
small-range channel, so it matters most where channel scales differ — common in depthwise convs.

**S6.** True `dq/dx = 0` almost everywhere (piecewise constant) and undefined at the steps. STE surrogate:
`dq/dx := 1` for `x` inside `[qmin·s, qmax·s]`, else `0`. The zero true gradient gives no learning signal; the
identity surrogate lets the loss push weights toward grid points that lower loss — biased but useful (Bengio
et al., 2013).

**S7.** `softmax(z/T)_i = e^{z_i/T}/Σe^{z_j/T}`; as `T→∞` exponents →0 so all `p_i→1/C` (uniform); at `T=1`
it is the plain softmax. Intermediate `T` inflates the relative mass on non-top classes, exposing the teacher's
similarity structure (dark knowledge). The gradient of the softened cross-entropy scales as `~1/T²`, so
multiplying the distillation term by `T²` keeps its magnitude comparable to the hard-label term.

**S8.** After (a): ~60 MB FP32 (0.6·100, roughly). After (b) INT8: ~1/4 → ~15 MB. Assumptions: prune removes
channels uniformly and quantization is ~4x on weights (ignore small overheads). Structured pruning shrinks the
real tensor dimensions, so dense kernels run faster; unstructured pruning leaves the dense shape (and kernel
time) unchanged absent sparse-compute support.

**S9.** Mean `= (19+20+20+21+19+55+20+22+20+21)/10 = 23.7`; sorted `[19,19,20,20,20,20,21,21,22,55]`, median
`= 20`; p90 (9th value) `= 22`, but the max `55` exceeds 33. By mean/median it "meets" 30 fps; by the tail a
`55 ms` frame drops. Report the tail for a real-time stream — a single dropped frame is a user-visible/safety
event; the `55` is a real periodic stall (thermal/GC/DVFS), not noise to average away.

**S10.** Example: model A is depthwise-heavy (low FLOPs, low arithmetic intensity → memory-bound, poor
utilization), model B is dense-conv-heavy (higher FLOPs but high intensity → compute-plateau, high
utilization); A can be slower in wall-clock despite fewer FLOPs. Or A uses an op the NPU lacks → CPU fallback.
Hence FLOPs mispredict latency; MnasNet/ProxylessNAS optimize *measured* on-device latency, the metric that
actually ships.
