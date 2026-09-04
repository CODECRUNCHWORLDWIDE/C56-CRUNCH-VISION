# Lecture 1 — Keypoints & pose: heatmap regression, sub-pixel decoding, and OKS

A **keypoint** is a specific, semantically named point to locate: a person's left elbow, the
corner of an eye, the tip of a surgical tool, a box corner for 6-DoF object pose. **Pose estimation**
finds a *set* of keypoints and their connections — for a human, a skeleton of joints. It is localization
at finer grain than boxes (Week 6) or masks (Week 7): exact coordinates for named points. This lecture
makes the dominant estimation paradigm rigorous.

## Why not regress coordinates directly?

The obvious approach — output `2K` numbers for `K` keypoints and regress them with an L2 loss — is
**DeepPose** (Toshev & Szegedy, 2014, CVPR), the first deep pose method. It works but trains poorly and
plateaus below the state of the art. The reason is structural: a fully-connected coordinate head must
collapse a spatial feature map to a global vector, discarding the translation-equivariant structure that
convolutions are good at; the mapping from pixels to a real-valued coordinate is highly nonlinear, and a
single scalar cannot represent *multimodality* (two plausible elbow locations) or *uncertainty*.

## Heatmap regression

The dominant approach (Tompson et al., 2014, NeurIPS; Newell et al., 2016, "Stacked Hourglass Networks",
ECCV) is **heatmap regression**. For each keypoint `k` the network — an encoder-decoder like Week 7 —
outputs a full-resolution **heatmap** `H_k`. The training target is not a one-hot pixel but a **rendered
2-D Gaussian** centred on the ground-truth location `(u_k, v_k)`:

    T_k(x, y) = exp( -[ (x - u_k)^2 + (y - v_k)^2 ] / (2 sigma^2) ).

The loss is pixel-wise MSE `sum_k || H_k - T_k ||^2` (or a per-pixel BCE). Rendering a Gaussian rather
than a delta does three things: (1) it turns a needle-in-a-haystack classification into a smooth, dense
regression that convolutions solve naturally; (2) the spread `sigma` injects tolerance — being one pixel
off is *almost* right, so gradients are informative everywhere near the joint; (3) the peak's sharpness
encodes **confidence**: a diffuse blob signals ambiguity, a tight blob certainty. HRNet (Sun et al., 2019,
"Deep High-Resolution Representation Learning", CVPR) pushes this further by maintaining high-resolution
representations throughout, which sharpens heatmaps and remains a strong top-down backbone.

## Decoding: argmax, soft-argmax, and DSNT

At inference you must turn a heatmap into a coordinate. The naive decoder is the **argmax** — the peak
pixel. But argmax has two problems. First, it is **non-differentiable**, so you cannot train coordinates
end-to-end through it. Second, it quantizes to integer pixels: on a heatmap downsampled 4x from the input,
argmax alone carries up to +/-2 input-pixel error, which is why heatmap methods add an empirical quarter-
offset toward the second-highest neighbour.

The principled fix is **soft-argmax** / **DSNT** (Differentiable Spatial to Numerical Transform; Nibali et
al., 2018). Normalize the heatmap to a probability map `p(x, y) = softmax(H)` and take the *expectation* of
the coordinate grid:

    u_hat = sum_{x,y} x * p(x, y),   v_hat = sum_{x,y} y * p(x, y).

This is differentiable, sub-pixel, and returns numerical coordinates you can supervise directly with an L2
loss on `(u_hat, v_hat)` — plus a regularizer keeping `p` tight. **Integral regression** (Sun et al., 2018,
"Integral Human Pose Regression", ECCV) is the same expectation trick and unifies heatmap and coordinate
views: you keep the heatmap's trainability *and* get a continuous coordinate with no quantization bias.

## From keypoints to pose, and the standard formats

Human pose = keypoints + a **skeleton** (which joints connect: shoulder-elbow-wrist). COCO (Lin et al.,
2014) defines 17 body keypoints; MPII uses 16. Drawing the skeleton turns a scatter into a readable
posture and lets you compute joint angles (the basis for the pose-analyzer challenge).

## Evaluating pose: PCK and OKS

Do not report pixel MSE. Two metrics dominate. **PCK** (Percentage of Correct Keypoints) counts a keypoint
correct if it lands within a threshold — often a fraction of head or torso size — of ground truth; it is
scale-relative but coarse. The COCO standard is **Object Keypoint Similarity (OKS)**, the keypoint analogue
of IoU:

    OKS = [ sum_i exp( -d_i^2 / (2 s^2 k_i^2) ) * 1[v_i > 0] ] / sum_i 1[v_i > 0],

where `d_i` is the Euclidean error of keypoint `i`, `s` is the object scale (sqrt area), `k_i` a per-keypoint
falloff constant (wrists are harder than eyes), and `v_i` the visibility flag (Ronchi & Perona, 2017,
"Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation", ICCV). OKS in `[0, 1]` feeds a mAP
computed exactly like detection AP but with OKS replacing box IoU. Report OKS-AP for people.

## Top-down vs. bottom-up (preview of Lecture 4)

For *multiple* people, **top-down** first detects each person (Week 6), then poses each crop
independently — accurate, but cost scales with the number of people and it inherits detector failures.
**Bottom-up** detects all keypoints at once, then *groups* them into individuals at constant cost — better
for crowds, but the grouping is the hard part. We derive the grouping mathematics in Lecture 4.

## Common pitfalls

- **Regressing coordinates directly and wondering why it plateaus.** Use heatmaps (or integral regression
  on heatmaps); the spatial prior is the whole point.
- **Decoding with bare argmax.** You inherit quantization error and cannot backprop coordinates. Soft-argmax
  / DSNT removes both problems.
- **Reporting pixel error instead of OKS.** OKS normalizes by object scale and per-joint difficulty; raw
  pixels are not comparable across image sizes or joints.

**Takeaway:** keypoint estimation renders each keypoint as a Gaussian heatmap and regresses it; the peak's
location is the coordinate and its sharpness is confidence. Decode sub-pixel with soft-argmax / DSNT to stay
differentiable and quantization-free, connect keypoints into a skeleton, and evaluate with OKS-AP, not pixel
error. The same machinery locates faces, hands, animals (DeepLabCut), and object corners for 6-DoF pose.
