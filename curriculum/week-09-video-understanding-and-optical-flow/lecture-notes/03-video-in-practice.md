# Lecture 3 — Video in practice: cost, data & honesty

Video is where vision meets hard engineering reality. The models are conceptually clean
extensions of image models, but the *practical* constraints — data volume, compute, memory, labeling,
and honest evaluation — dominate every real video project. Respecting them is the difference between a
demo and a system.

## Video is expensive, everywhere

- **Data volume.** One second of 30 fps video is 30 images. A modest dataset is *millions* of frames —
  huge to store, decode, and process. I/O (video decoding) often bottlenecks training harder than the
  GPU. Efficient decoders (NVIDIA DALI, decord) exist precisely because `cv2.VideoCapture` in a Python
  loop cannot keep a GPU fed.
- **Compute & memory.** Processing frame stacks (3D convs, spacetime attention) multiplies image cost
  by the temporal dimension `T`. Memory bounds how many frames fit at once, forcing short clips or
  frame subsampling and small batch sizes.
- **Labeling.** Annotating video (per-frame boxes, action start/end boundaries) is brutally
  labor-intensive — far worse than images, and boundaries are genuinely ambiguous (when does 'standing
  up' begin?). This scarcity makes pretraining (Kinetics) and transfer learning essential.

## The practical toolkit

- **Sample frames; do not use all.** Most video models process a handful of sampled frames per clip
  (e.g. 8-32), not every frame. Sampling strategy is a real design decision: *uniform* sampling covers
  the whole clip; TSN-style (Wang et al. 2016) segment-and-sample reduces variance; dense sampling
  captures fast motion. SlowFast (Feichtenhofer et al. 2019) runs two pathways at different frame
  rates on purpose.
- **Clip-based training, video-level inference.** Train on short fixed-length clips sampled from longer
  videos; at test time aggregate predictions over multiple clips (and spatial crops) for a full-video
  label.
- **Start pretrained.** Kinetics-pretrained backbones then fine-tuned — the Week-5 lesson, now
  essential because video data and compute are scarce.
- **Lower resolution / fewer frames** are the first knobs when you hit a memory or latency wall (a
  Week-11 edge preview).

## Evaluating video honestly

- **Split by *video*, not frame.** Frames within one video are near-duplicates; a frame-level split
  leaks the test set into training and inflates accuracy — the group-split lesson (Week 4) with the
  highest stakes yet. Whole videos go entirely in one split.
- **The single-frame baseline is a truth serum.** Always compare a temporal model to a
  frame-aggregation baseline. If your expensive 3D or Transformer model barely beats it, the *task* did
  not need temporal modeling — an honest, money-saving finding and a guard against over-engineering.
- **Temporal metrics.** For temporal action *localization* (when did the action start/end?), use
  **temporal IoU** — the 1-D analogue of spatial IoU — and mean average precision over tIoU thresholds
  (the activitynet / THUMOS protocol).
- **Watch real-footage failure.** Camera motion, lighting changes, motion blur, compression artifacts,
  and unusual viewpoints break video models — and the failures are visible on playback, so watch it.

## Honesty and ethics — this one is heavy

Video recognition is powerful **surveillance** technology. Action recognition on people, gait analysis,
and behavior monitoring carry the same — heightened — privacy weight as tracking (Week 8), plus the
temporal dimension makes re-identification and behavioral profiling easier. The course's rules stand
and tighten here:

- **Consent and lawful purpose.** Recording and analyzing people's behavior demands a lawful basis and,
  wherever feasible, informed consent — not a scraped dataset of strangers.
- **Dataset bias is structural.** Kinetics and similar sets skew heavily by geography, language, and
  demographics (Western, English-labeled, able-bodied). A model trained on them *will* underperform on
  under-represented groups; report this, do not hide it.
- **Refuse the mirror test.** Do not build behavior surveillance you would not accept aimed at yourself
  and the people you love.
- **Label synthetic video as synthetic.** Generated or manipulated footage must be disclosed.

**Takeaway:** video's hard part is engineering reality, not architecture — data volume, compute, memory,
and labeling all multiply by time. Sample frames, train on clips, start from Kinetics-pretrained models,
and *always* compare to a single-frame baseline to check the temporal cost was earned. Split by video to
avoid leakage, use temporal IoU for localization, watch real-footage failure, and treat behavior
recognition on people with heightened, consent-first privacy care.
