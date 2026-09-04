# Lecture 2 — Architectures for video understanding

A single-frame CNN throws away time. Video understanding — recognizing *actions* and events,
not just objects — needs architectures that reason across frames. There is a **ladder** of approaches,
each adding more temporal modeling at more compute and memory cost. Knowing where to stand on that
ladder for a given task and budget is the central engineering judgment of video.

## Why a single frame is not enough

Many actions are defined by *change over time*: 'opening' vs. 'closing' a door, 'sitting down' vs.
'standing up', 'picking up' vs. 'putting down'. A single frame is ambiguous or identical between them.
The signal lives in the **order and motion** of frames, so the model must see multiple frames and model
their temporal relationship — not just pool per-frame appearance.

## The ladder of temporal modeling

**Rung 1 — Frame-level + late aggregation.** Run a 2D CNN on each sampled frame independently, then
pool predictions (average, or a small temporal head). Cheap, and a *surprisingly strong baseline*:
for actions a single frame nearly reveals ('playing guitar', 'swimming'), appearance alone wins. But
it is blind to fine motion and order. Karpathy et al. (2014) showed early- vs. late-fusion variants
barely beat single-frame on Sports-1M — a humbling result that motivated everything above.

**Rung 2 — 3D CNNs.** Extend convolution to *time*: a 3D kernel of shape `(t, h, w)` slides over a
stack of frames, learning spatiotemporal features directly (C3D, Tran et al. 2015; I3D, Carreira &
Zisserman 2017, which "inflates" 2D ImageNet kernels into 3D to bootstrap from image pretraining).
Powerful, but expensive: a `3×3×3` kernel has 27 weights per channel pair vs. 9 for `3×3`, and the
activation tensor carries an extra temporal axis, multiplying both FLOPs and memory.

**Rung 3 — Two-stream networks.** Two parallel CNNs (Simonyan & Zisserman 2014): a **spatial stream**
on RGB frames (appearance — *what* objects) and a **temporal stream** on stacked **optical flow**
(motion — *how* things move), fused at the end. Explicitly feeding flow (Lecture 1) was a breakthrough
because the network does not have to learn motion from raw pixels — but it needs a flow algorithm as a
preprocessing step, which is itself expensive.

**Rung 4 — (2+1)D / factorized convolution.** Split a 3D convolution into a 2D **spatial** conv
followed by a 1D **temporal** conv (Tran et al. 2018, "R(2+1)D"). This has nearly the expressive power
of full 3D at much lower cost, adds an extra nonlinearity between the two, and is easier to optimize.
A practical sweet spot — often the default when you need real temporal modeling without a Transformer.

**Rung 5 — Video Transformers.** Extend the Vision Transformer (Week 10) to spacetime: TimeSformer
(Bertasius et al. 2021), ViViT (Arnab et al. 2021), Video Swin (Liu et al. 2022) apply self-attention
across space *and* time. Attention models long-range temporal dependencies (an action spanning many
frames) that convolution captures only locally — now state of the art, at a steep compute price
because attention is quadratic in the number of spacetime tokens (Lecture 5 addresses the fix).

```python
import torchvision
# R(2+1)D-18 pretrained on Kinetics-400
model = torchvision.models.video.r2plus1d_18(weights="KINETICS400_V1").eval()
# input: (batch, channels=3, frames=T, H, W)
```

## The recurring trade-off

More temporal modeling → better on motion-heavy actions → more compute, memory, and data. The right
rung depends on the task (does it need *fine* motion, or is appearance enough?), the budget, and the
available labeled video. A frame-aggregation baseline is *always* worth trying first — sometimes the
action is mostly visible in one frame and you save enormously (Lecture 3 makes this a discipline).

**Takeaway:** video understanding needs temporal modeling, and there is a ladder — frame aggregation
(cheap baseline) → 3D CNN (spatiotemporal kernels, expensive) → two-stream (RGB + optical-flow motion)
→ (2+1)D (factorized, efficient sweet spot) → spacetime Transformer (SOTA, quadratic cost). Each rung
adds temporal power at more cost. Climb only as far as the task demands, and start from the cheap
baseline.
