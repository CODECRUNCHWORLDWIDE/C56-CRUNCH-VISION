# Week 9 — Graduate Problem Set: Flow, Structure Tensors, Architectures, and Attention Cost

Ten problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution
sketches are at the end — attempt each fully first. Notation: `I(x,y,t)` is image intensity,
`∇I=(I_x,I_y)` the spatial gradient, `I_t` the temporal gradient, `(u,v)` the flow, and `M` the 2x2
structure tensor `Σ_W [[I_x², I_xI_y],[I_xI_y, I_y²]]`.

**P1 (flow constraint derivation).** Starting from brightness constancy
`I(x,y,t)=I(x+u·dt, y+v·dt, t+dt)`, derive the optical flow constraint `I_x·u + I_y·v + I_t = 0` via a
first-order Taylor expansion. State clearly which approximation you made and when it fails.

**P2 (normal flow).** Show that the flow constraint determines only the flow component **parallel to
`∇I`**, and give its magnitude `|I_t|/‖∇I‖`. Explain in one sentence why this *is* the aperture problem.

**P3 (structure-tensor conditioning).** For Lucas-Kanade, the least-squares solution is
`(u,v) = M⁻¹ b`. Using the eigenvalues `λ₁ ≥ λ₂` of the symmetric PSD matrix `M`, state the condition
for a reliable solution, and describe the geometry (flat region / edge / corner) corresponding to
(a) `λ₁ ≈ λ₂ ≈ 0`, (b) `λ₁ ≫ λ₂ ≈ 0`, (c) `λ₁ ≈ λ₂ ≫ 0`.

**P4 (Horn-Schunck energy).** Write the Horn-Schunck energy functional and derive the form of its
Euler-Lagrange equations (you may leave the Laplacian symbolic). Explain physically what role the
smoothness weight `α` plays and what happens as `α → 0` and `α → ∞`.

**P5 (warping and coarse-to-fine).** Explain why warping frame 2 toward frame 1 with a current flow
estimate lets a *small* search window handle *large* motion. Why can a coarse-to-fine pyramid still miss
a small, fast-moving object, and how does RAFT's all-pairs correlation volume avoid that failure?

**P6 (3D conv FLOP count).** A 3D convolution has kernel `(k_t, k_h, k_w)`, `C_in` input and `C_out`
output channels, applied to an output volume of size `T'×H'×W'`. Write the multiply-add count. Compare
it to a `(2+1)D` factorization (a `1×k_h×k_w` spatial conv into `M` channels, then `k_t×1×1` temporal
conv) and state the rough cost ratio for `k_t=k_h=k_w=3`, ignoring the intermediate-channel choice.

**P7 (attention cost).** For a video with `N` spatial tokens per frame and `T` frames, give the
self-attention cost of (a) joint space-time attention and (b) TimeSformer's divided space-time
attention. For `N=196`, `T=8`, compute the ratio of the two, and state which axis (resolution or clip
length) you would grow first if you had to.

**P8 (leakage).** A team reports 96% action-recognition accuracy but split their dataset by **frame**.
Explain quantitatively why this number is meaningless, estimate the direction and rough scale of the
inflation, and describe the correct by-video protocol.

**P9 (baseline value of information).** You have a single-frame baseline at 71% and a 3D CNN at 74% on a
video-split test set of 500 clips, at 40x the inference cost. Treating accuracy differences with a
back-of-envelope binomial standard error, is the 3-point gap likely real, and would you deploy the 3D
model for a latency-sensitive edge camera? Argue explicitly.

**P10 (open analysis).** VideoMAE masks ~90-95% of spacetime patches, far more than image MAE's ~75%.
Argue from the information content and temporal redundancy of video why the optimal mask ratio is higher
for video, what would go wrong at a 50% ratio, and how this connects to the choice between contrastive
and reconstructive self-supervision. (Open-ended; reason carefully rather than guess.)

---

## Solution sketches

**S1.** Expand the RHS: `I + I_x·u·dt + I_y·v·dt + I_t·dt + O(dt²)`. Cancel `I`, divide by `dt`, drop
higher-order terms → `I_x·u + I_y·v + I_t = 0`. The first-order (small-displacement) approximation fails
for large/fast motion — hence pyramids.

**S2.** The constraint is `∇I·(u,v) = -I_t`. Decompose `(u,v)` into components along and perpendicular
to `∇I`; the perpendicular (edge-parallel) part contributes 0. So only the parallel part is fixed, with
magnitude `|I_t|/‖∇I‖`. Edge-parallel motion is unconstrained = aperture problem.

**S3.** Reliable iff both eigenvalues are large and `λ₂` bounded away from 0 (small condition number).
(a) both ~0 → flat, textureless, no flow signal; (b) `λ₁≫λ₂≈0` → edge, aperture problem, only normal
flow; (c) both large → corner, `M` well-conditioned, full flow recoverable.

**S4.** `E(u,v)=∫∫ (I_xu+I_yv+I_t)² + α²(‖∇u‖²+‖∇v‖²) dxdy`. Euler-Lagrange:
`I_x(I_xu+I_yv+I_t) = α²Δu` and `I_y(I_xu+I_yv+I_t) = α²Δv` (Δ = Laplacian). Large `α` → very smooth,
over-smoothed across motion boundaries; `α→0` → per-pixel underdetermined (back to the aperture
problem). `α` trades data fidelity vs. smoothness.

**S5.** Warping removes the bulk motion so only the small residual remains, which a small window/Taylor
expansion can capture. A small fast object can be blurred away or missed at the coarse pyramid level and
never recovered on refinement. RAFT computes correlation between *all* pixel pairs at full resolution
once, so matching cost for *any* displacement (large or small) is available to the recurrent updater —
nothing is lost to downsampling.

**S6.** 3D conv MACs ≈ `C_in·C_out·k_t·k_h·k_w·T'·H'·W'`. With `3³=27` kernel volume. (2+1)D:
spatial `C_in·M·k_h·k_w·(·)` + temporal `M·C_out·k_t·(·)` ≈ `9 + 3` weight-volume vs `27` (choosing
`M≈C`), so roughly `12/27 ≈ 0.44` — under half the cost, plus an added nonlinearity.

**S7.** (a) joint: `O((N·T)²·d)`. (b) divided: spatial `O(T·N²·d)` + temporal `O(N·T²·d)`. Ratio
joint/divided ≈ `(NT)² / (TN² + NT²) = NT/(N+T)`. For `N=196,T=8`: `196·8/(196+8)=1568/204≈7.7×` cheaper.
Grow clip length `T` first — it is the cheaper axis (temporal cost `T²` with small `T`) and time is
where the extra information is; resolution grows `N` which dominates spatial `N²`.

**S8.** Frames within a video are near-duplicates; a frame split puts near-copies of test frames in
training, so the model memorizes rather than generalizes — accuracy approaches training accuracy and can
be inflated by tens of points (often the true video-split number is far lower). Correct: assign each
*whole video* to exactly one split, so no frame from a test video is ever seen in training.

**S9.** Binomial SE ≈ `√(p(1-p)/n) = √(0.72·0.28/500) ≈ 0.020` = 2 points per model; the SE of a
*difference* is ~`√2·0.02 ≈ 0.028`, so a 3-point gap is only ~1σ — **not clearly significant** on 500
clips. At 40× cost for a latency-sensitive edge camera, deploy the single-frame baseline (or a cheap
temporal model); the gap does not justify 40× latency/energy.

**S10.** Video's per-token information is low because adjacent frames are near-redundant; at a low mask
ratio the model reconstructs masked patches by copying temporal neighbors (trivial), learning nothing.
A ~90-95% ratio removes that shortcut, forcing genuine spatiotemporal inference. At 50%, reconstruction
is too easy → weak features. This favors reconstructive MAE for video (redundancy makes high masking a
strong pretext) while contrastive methods instead rely on temporal augmentations to build invariance;
both exploit temporal coherence as free supervision, differently.
