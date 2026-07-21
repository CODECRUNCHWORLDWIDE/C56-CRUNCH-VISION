# Lecture 3 — Video in practice: cost, data & honesty

Video is where vision meets hard engineering reality. The models are conceptually straightforward extensions of image models, but the *practical* constraints — data volume, compute, memory, and evaluation — dominate every real video project. Respecting them is the difference between a demo and a system.

## Video is expensive, everywhere

- **Data volume.** One second of 30 fps video is 30 images. A short dataset is millions of frames — huge to store, load, and process. I/O often bottlenecks training more than the GPU.
- **Compute & memory.** Processing frame stacks (3D convs, spacetime attention) multiplies image costs by the temporal dimension. Memory limits how many frames you can hold at once, forcing short clips or frame subsampling.
- **Labeling.** Annotating video (per-frame boxes, action boundaries) is brutally labor-intensive — far worse than images. This scarcity shapes what's feasible, and makes pretraining (Kinetics-pretrained models) and transfer learning essential.

## The practical toolkit

- **Sample frames, don't use all.** Most video models process a handful of sampled frames (e.g. 8–32) per clip, not every frame. Choosing the sampling (uniform? dense around motion?) is a real design decision.
- **Clip-based training.** Train on short fixed-length clips sampled from longer videos; aggregate clip predictions at inference for a full video.
- **Start pretrained.** Use models pretrained on large video datasets (Kinetics) and fine-tune — the Week-5 lesson, now essential because video data and compute are scarce.
- **Lower resolution / fewer frames** are the first knobs when you hit memory or latency limits (a Week-11 edge preview).

```mermaid
flowchart LR
  A["Sample eight to thirty two frames per clip"] --> B["Train on short fixed length clips"]
  B --> C["Start from Kinetics pretrained model"]
  C --> D["Lower resolution or fewer frames if memory limited"]
```
*The practical toolkit for training video models under data, compute, and memory limits.*

## Evaluating video honestly

- **Held-out *videos*, not frames.** Frames from the same video are highly correlated; splitting by frame leaks. Split by video (the group-split lesson, Week 4).
- **The single-frame baseline is a truth serum.** Always compare your temporal model to a frame-aggregation baseline. If your expensive 3D model barely beats it, the task didn't need temporal modeling — an honest, money-saving finding, and a check against over-engineering.
- **Temporal metrics** for tasks like temporal action *localization* (when did the action start/end?) use temporal IoU, analogous to spatial IoU.
- **Watch failure on real footage** — camera motion, lighting changes, and unusual viewpoints break video models, and the failures are visible on playback.

## Honesty and ethics

Video recognition is powerful surveillance technology. Action recognition on people, gait analysis, and behavior monitoring carry the same — heightened — privacy weight as tracking (Week 8). The course's rules stand: consent, lawful purpose, honest reporting of bias (video datasets skew heavily by geography and demographics), and refusing to build surveillance you wouldn't accept aimed at yourself. And label synthetic or generated video as synthetic.

**Takeaway:** video's hard part is engineering reality, not architecture — data volume, compute, memory, and labeling all multiply by time. Sample frames, train on clips, start from video-pretrained models, and *always* compare to a single-frame baseline to check the temporal cost was worth it. Split by video to avoid leakage, watch real-footage failures, and treat behavior recognition with heightened privacy care.
