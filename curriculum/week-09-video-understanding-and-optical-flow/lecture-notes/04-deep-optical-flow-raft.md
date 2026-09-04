# Lecture 4 — Deep optical flow: correlation volumes, warping, and RAFT

Classical flow (Lecture 1) hand-designs its prior: local constancy (Lucas-Kanade) or global
smoothness (Horn-Schunck). Both struggle with **large motion**, **occlusion**, and **textureless
regions**, and both need coarse-to-fine pyramids that lose small fast objects. Since roughly 2015,
learned flow has replaced the hand-designed prior with a network trained end-to-end on ground-truth
flow. This lecture builds the two ideas that define the modern state of the art: the **correlation
volume** and the **recurrent update operator**.

## Why learn flow at all?

Ground-truth flow is nearly impossible to label by hand (you cannot ask an annotator for a subpixel
motion vector at every pixel), so learned flow is trained on **synthetic** data with perfect labels —
FlyingChairs and FlyingThings3D (Mayer et al. 2016), and the MPI Sintel benchmark (Butler et al. 2012)
rendered from an open film — then fine-tuned on smaller real sets (KITTI). The bet: features learned to
predict synthetic flow transfer to real footage. It largely holds, which is itself a lesson about
synthetic-to-real transfer.

## FlowNet and the correlation layer

FlowNet (Dosovitskiy et al. 2015) framed flow as a supervised regression: encode both frames with a
CNN, then decode a dense flow field. Its key module is the **correlation layer**, which computes, for
each location in frame 1, the dot-product similarity with candidate locations in frame 2 over a search
window. Correlation *is* the learned analogue of brightness matching — high correlation means "this is
where the patch went." But a fixed search window bounds the maximum displacement.

## PWC-Net: pyramids, warping, cost volume

PWC-Net (Sun et al. 2018) folds three classical ideas into a compact network and dominated its era:

- **P**yramid — build a learned feature pyramid of each frame.
- **W**arping — at each level, warp frame-2 features toward frame-1 using the current (upsampled) flow
  estimate, so only the small *residual* motion remains — exactly Lecture 1's coarse-to-fine warping,
  now on learned features.
- **C**ost volume — compute a *local* correlation between frame-1 features and warped frame-2 features.

Warping lets a *small* correlation window handle *large* motion, because each pyramid level only has to
find the leftover displacement. This is efficient but inherits the pyramid's weakness: motion missed at
a coarse level cannot be recovered.

## RAFT: all-pairs correlation + recurrent refinement

RAFT (Teed & Deng, ECCV 2020, best-paper) is the current reference design and changed how flow is done:

1. **All-pairs 4-D correlation volume.** Instead of a local, warped cost volume, RAFT computes the
   correlation between *every* pixel in frame 1 and *every* pixel in frame 2, once, forming a 4-D volume
   `C[i, j, k, l]`. Pooling it at several scales gives a pyramid that stores matching costs for *all*
   displacements — large and small — up front, so nothing is lost to a coarse level.
2. **Recurrent GRU update.** A convolutional GRU starts from zero flow and, at each step, *looks up* the
   correlation values at the current flow estimate, combines them with context features, and predicts a
   **residual** flow update `Δf`. Iterating ~12-32 times refines the field — a *learned optimizer* that
   mimics first-order iterative refinement (echoing Horn-Schunck's Gauss-Seidel sweeps, but learned).
3. **Single high-resolution field, shared weights.** All updates share weights and operate at a fixed
   high resolution — no lossy coarse-to-fine cascade.

RAFT set the accuracy standard on Sintel and KITTI, generalizes well, and its recurrent-lookup design
seeded successors (GMA for occlusion, FlowFormer replacing the GRU with a Transformer). The intellectual
arc is clean: brightness constancy → correlation as learned matching → all-pairs volume so no
displacement is lost → a learned recurrent update replacing the hand-tuned smoothness prior.

## Occlusion and the limits

No flow is defined for pixels that are **occluded** (visible in one frame, hidden in the other) — there
is no true match. Learned models *hallucinate* plausible flow there from context, which is usually
desirable but is a guess, not a measurement. Brightness constancy still ultimately underlies the
training signal, so specular highlights, transparency, and drastic lighting changes remain hard.

**Takeaway:** modern flow replaces hand-designed priors with learning: a **correlation volume** turns
brightness matching into a differentiable similarity search, PWC-Net makes it efficient via feature
pyramids and warping, and **RAFT** (Teed & Deng 2020) computes an **all-pairs** correlation volume once
and refines flow with a **recurrent GRU** that acts as a learned iterative optimizer — winning on large
motion and occlusion where classical methods fail. Trained on synthetic data (Sintel, FlyingThings),
it transfers to real footage, but occluded and non-Lambertian pixels remain fundamentally ambiguous.
