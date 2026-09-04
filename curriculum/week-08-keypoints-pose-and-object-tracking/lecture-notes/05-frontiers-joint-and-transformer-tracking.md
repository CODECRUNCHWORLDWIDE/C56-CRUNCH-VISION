# Lecture 5 — Frontiers: joint detection-embedding, ByteTrack, and transformer trackers

Classic tracking-by-detection (Lectures 2 and 4) runs detection, embedding, filtering, and
assignment as *separate* stages. The last five years collapsed these into fewer, learned components. This
lecture surveys the state of the art so you can read modern MOT papers and choose an architecture for the
mini-project deliberately.

## Separate detection and embedding (SDE) vs. joint (JDE)

Deep SORT is an **SDE** tracker: a detector, then a *separate* re-ID network embeds each crop. Accurate but
slow — two forward passes per frame. **JDE** ("Towards Real-Time Multi-Object Tracking", Wang et al., 2020,
ECCV) makes the network output detections *and* appearance embeddings from one backbone in a single pass, a
**joint detection and embedding** design that hits real-time speeds. **FairMOT** (Zhang et al., 2021, IJCV)
diagnosed why naive JDE underperforms: the anchor-based detection head and the embedding head *compete* —
anchors are box-centric and coarse, embeddings need pixel-precise, high-resolution features. FairMOT fixes
this with an **anchor-free**, point-based detection head (CenterNet-style) and a parallel, equally-weighted
embedding head on high-resolution features, treating the two tasks *fairly* — hence the name. The lesson
generalizes: multi-task heads that need incompatible feature geometries fight, and the fix is to align their
receptive requirements.

## ByteTrack: use the low-score detections

**ByteTrack** (Zhang et al., 2022, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
ECCV) is deceptively simple and near-SOTA *without any appearance model*. The standard pipeline discards
low-confidence detections (say score < 0.5) as false positives. But a low score often means a *real object
that is occluded or blurred* — exactly the frames where tracks are most likely to be lost. ByteTrack keeps
every box and associates in **two passes**: (1) match tracks to high-score detections first (as usual);
(2) match the *remaining, unmatched* tracks to the *low-score* detections, using the tracks' motion
prediction to disambiguate. A leftover low-score box that overlaps a predicted track is very likely the same
occluded object; a leftover box that matches nothing is discarded. This recovers occluded objects, slashes
fragmentation, and topped MOT17/MOT20 leaderboards with a plain Kalman-plus-IoU core. It is the strongest
argument that *association strategy*, not a fancier embedding, is often the highest-leverage improvement.

## Transformer trackers: tracks as persistent queries

DETR (Carion et al., 2020, Week 6) recast detection as set prediction with learned object *queries*. The
transformer trackers extend this to time. **TrackFormer** (Meinhardt et al., 2022, CVPR) and **MOTR** (Zeng
et al., 2022, ECCV) introduce **track queries**: the object queries that produced detections in frame `t`
are *carried forward* as queries into frame `t+1`. A track query attends to the new frame's features and
directly regresses the same object's new box — so association is performed *implicitly by attention*, with
no explicit Hungarian step at inference and no separate motion filter. New objects are picked up by fresh
detection queries; disappearing objects produce a "no-object" class and their query is retired.
**TransTrack** (Sun et al., 2020) is a related design mixing learned and detection queries. The appeal is a
single end-to-end network with no hand-tuned association; the cost is heavy training data, careful handling
of long occlusions (a retired query cannot trivially re-attach), and compute. This "tracking-by-attention"
paradigm (contrast: tracking-by-detection) is the current research frontier for MOT.

## Frontiers in pose, briefly

Pose is moving the same way. **Bottom-up grouping** — the hard step from Lecture 1 — has two canonical
solutions worth knowing precisely. **Part-Affinity Fields** (OpenPose; Cao et al., 2017, CVPR) predict, for
each limb, a 2-D vector field pointing from one joint to the next; the association score of a candidate
limb is the **line integral** of that field along the segment joining two detected keypoints, and grouping
maximizes total limb score — a bipartite matching per limb type. **Associative Embedding** (Newell et al.,
2017, NeurIPS) instead has the network emit a scalar "tag" per keypoint and trains keypoints of the same
person to share a tag; grouping clusters by tag. On the estimation side, **ViTPose** (Xu et al., 2022,
NeurIPS) shows a plain Vision Transformer backbone with a simple decoder sets the top-down state of the art,
and **integral/soft-argmax regression** (Lecture 1) is standard for differentiable, quantization-free
coordinates. Beyond 2-D, **3-D lifting** and multi-view methods recover metric pose for biomechanics, and
**DeepLabCut** (Mathis et al., 2018, *Nature Neuroscience*) brought the same heatmap machinery to animal
behaviour, where it is now a scientific instrument.

## Choosing, honestly

There is no universal best tracker. For a clean, few-object clip, SORT is enough. For crossings and
occlusion, add appearance (Deep SORT) or the two-pass association of ByteTrack. For real-time on-device,
prefer a joint JDE/FairMOT design. For research or heavily-occluded crowds with training budget, transformer
trackers lead. State your constraints — object count, occlusion, latency, compute, data — and pick to match;
"use the newest paper" is not an engineering argument.

## Common pitfalls

- **Reaching for a transformer tracker on a small clip.** They need large training sets and compute;
  ByteTrack or Deep SORT will beat them out of the box on modest data.
- **Assuming appearance is always needed.** ByteTrack shows motion + smart association can match appearance-
  based trackers; profile before adding a re-ID network.
- **Ignoring domain shift.** Re-ID embeddings and learned trackers trained on pedestrians transfer poorly to
  vehicles, animals, or a new camera. Validate on *your* distribution.

**Takeaway:** modern MOT collapses the classic stages — JDE/FairMOT fuse detection and embedding in one pass
(fixing the head-competition problem), ByteTrack shows that associating *every* box, including low-score
ones, in two passes rivals appearance-based trackers, and transformer trackers (TrackFormer, MOTR) make
association implicit by carrying track queries through time. Pose is on the same trajectory: PAF line
integrals and associative embeddings for grouping, ViTPose and integral regression for estimation. Choose by
your constraints, not by novelty.
