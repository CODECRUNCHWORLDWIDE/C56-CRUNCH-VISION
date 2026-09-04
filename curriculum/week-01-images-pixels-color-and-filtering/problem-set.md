# Week 1 — Graduate Problem Set: Sampling, Color, Convolution, and Frequency

Twelve problems, easy to hard, mixing derivation, proof, computation, and open analysis.
Solution sketches are at the end — attempt each fully before reading them. Notation: `f[m,n]` is a discrete
image, `h` a kernel, `*` convolution, `F, H` their 2-D DFTs, `G_σ` a Gaussian of standard deviation `σ`.

**P1 (formation & quantization).** An 8-bit sensor has a full-well capacity giving a maximum linear signal of
`L_max`. If you quantize *linearly* to 8 bits, how many code values fall in the top perceptual stop (the
brightest factor-of-2 of light) versus the bottom stop? Use this to explain, in two sentences, why sRGB
gamma-encodes before quantizing.

**P2 (channel layout & strides).** A color image is stored as a C-contiguous `(H, W, 3)` uint8 array. Give
the buffer offset of pixel `(y, x)` channel `c`. If a library instead wants `(3, H, W)` (CHW), is converting a
transpose (a view) or a copy? Explain via strides.

**P3 (grayscale as projection).** The luma weights `(0.299, 0.587, 0.114)` sum to 1. Why must luminance
weights sum to 1? What would happen to a neutral gray `(v, v, v)` if they summed to 1.2? Relate the weights to
the `Y` row of an RGB→XYZ matrix.

**P4 (HSV degeneracy).** Show that hue is undefined when `R = G = B`, and that saturation → 0 there. Give the
practical consequence for thresholding hue on near-gray or near-black pixels, and one way to guard against it.

**P5 (LSI ⇒ convolution).** Argue that a linear, shift-invariant operator `T` is determined entirely by its
response `h` to a single impulse `δ`, and that `T(f) = f * h`. (Hint: write `f` as a sum of shifted, scaled
impulses and apply linearity then shift-invariance.)

**P6 (convolution vs. correlation).** Convolution flips the kernel; correlation does not. Prove that for a
kernel symmetric under 180° rotation (`h[i,j] = h[-i,-j]`) the two operations are identical. Give a `2×2`
kernel for which they differ, with the two outputs on a small input.

**P7 (separability & rank).** Prove that a 2-D kernel is separable (`h = a bᵀ`, computable as two 1-D passes)
**iff** the kernel matrix has rank 1. Show the Gaussian is separable and the standard `3×3` Sobel
`[[-1,0,1],[-2,0,2],[-1,0,1]]` is separable; give its two 1-D factors.

**P8 (multiply counts).** For a `k×k` kernel on an `H×W` image, give the multiply count for (a) direct 2-D
convolution, (b) separable two-pass convolution, (c) FFT-based convolution. State the `k` at which (b) beats
(a) and, asymptotically in image size, why (c) wins for large kernels.

**P9 (convolution theorem).** State and justify `f * h ⇔ F · H`. Use it to explain (i) why cascading two
Gaussian blurs `G_{σ1} * G_{σ2}` equals a single `G_σ` with `σ² = σ1² + σ2²`, and (ii) why a box filter
produces ringing while a Gaussian does not (compare their frequency responses).

**P10 (sampling & aliasing).** A 1-D signal `cos(2π f₀ t)` is sampled at rate `f_s < 2f₀`. Show the samples
are identical to those of a lower-frequency cosine, and give that alias frequency. Extend the idea to explain
moiré on a photographed picket fence and why a Gaussian pre-blur before 2× downsampling removes it. State the
new Nyquist frequency after 2× decimation.

**P11 (median vs. mean).** For a neighborhood of intensities, prove the median minimizes the sum of absolute
deviations while the mean minimizes the sum of squared deviations. Use this to explain, precisely, why the
median filter annihilates salt-and-pepper (impulse) noise that a Gaussian blur merely spreads.

**P12 (open — scale space).** Argue why the Gaussian is a natural (indeed, under mild axioms, the unique)
smoothing kernel for building a scale space: as `σ` grows, no *new* local extrema should appear (structure
should only simplify). Relate this to why difference-of-Gaussians approximates the Laplacian-of-Gaussian and
serves as a blob detector (previewing SIFT). Open-ended: argue carefully rather than guess.

---

## Solution sketches

**S1.** Linear quantization spends half its 256 codes (128) on the top stop (light `L_max/2`..`L_max`) and
only `128/2^k` on the k-th stop down, so the darkest stops get a handful of codes → banding in shadows. Gamma
encoding (~V^(1/2.2)) redistributes codes roughly perceptually-uniformly (Weber's law), giving dark tones
enough levels — hence encode-then-quantize.

**S2.** Offset `= (y·W + x)·3 + c`. CHW is a **copy** in general: the fastest-varying axis changes from
channel to width, so no single stride relabeling yields it from an HWC contiguous buffer (a transpose gives a
non-contiguous *view*; materializing CHW-contiguous data copies).

**S3.** Weights must sum to 1 so a neutral gray maps to the same gray (`0.299v+0.587v+0.114v = v`); summing to
1.2 would brighten grays by 20% and shift white point. The weights are (approximately) the middle `Y` row of a
Rec.601 RGB→XYZ matrix — luminance is a fixed linear combination of primaries.

**S4.** With `R=G=B`, `max=min` so chroma `= max−min = 0` ⇒ saturation `= 0` and hue `= 0/0`, undefined.
Practically, hue is noise for gray/dark pixels, so hue thresholds must be gated by a minimum saturation and
value (as `cv2.inRange` bounds do).

**S5.** Write `f = Σ_{k,l} f[k,l]·δ[·−k, ·−l]`. Linearity: `T(f) = Σ f[k,l] T(δ[·−k,·−l])`. Shift-invariance:
`T(δ[·−k,·−l]) = h[·−k,·−l]` where `h = T(δ)`. So `T(f)[m,n] = Σ f[k,l] h[m−k,n−l] = (f*h)[m,n]`.

**S6.** Convolution uses `h[m−i,n−j]`; correlation `h[m+i,n+j]`. If `h[i,j]=h[−i,−j]` the index sets coincide,
so outputs match. Asymmetric example: `h=[[1,2],[3,4]]`; on an impulse the two outputs are 180°-rotated copies
of `h`, hence different.

**S7.** `h = a bᵀ` is exactly a rank-1 matrix; then `(f*h)[m,n] = Σ_i a[i] (Σ_j f[m−i,n−j] b[j])` — inner sum
is a 1-D conv with `b` along columns, outer with `a` along rows. Gaussian: `a[i]=exp(−i²/2σ²)`,
`b[j]=exp(−j²/2σ²)`. Sobel factors as `[1,2,1]ᵀ` (smoothing) × `[−1,0,1]` (difference), rank 1.

**S8.** (a) `HW·k²`; (b) `HW·2k`; (c) `O(HW·log(HW))` for the FFTs plus `HW` pointwise multiplies. (b) beats
(a) when `2k < k²` ⇒ `k > 2` (any real kernel). (c) wins for large `k` because its cost is independent of `k`
(after padding), while direct grows as `k²`.

**S9.** DFT diagonalizes convolution: shift-invariance means sinusoids are eigenfunctions, so `f*h` scales
each frequency by `H`. (i) Gaussians multiply to a Gaussian in frequency; product of two Gaussians `⇔` sum of
variances, so `σ²=σ1²+σ2²`. (ii) Box's response is a sinc with side-lobes (ringing); Gaussian's response is a
positive Gaussian, monotone roll-off, no side-lobes — no ringing.

**S10.** With `f_s<2f₀`, `cos(2π f₀ nΔ)=cos(2π(f₀−f_s)nΔ)` at the sample points, so the alias frequency is
`|f₀−f_s|` (folded into `[0,f_s/2]`). A fence's stripe frequency beats against the pixel grid → low-frequency
moiré. Pre-blur removes energy above the post-decimation Nyquist `f_s/4` (new rate `f_s/2`), so nothing folds.

**S11.** `d/dc Σ|y_i−c|` = `#{y_i<c} − #{y_i>c}`, zero at a median; `d/dc Σ(y_i−c)²` = `−2Σ(y_i−c)`, zero at the
mean. An impulse is one extreme `y_i`: it shifts the mean a lot but cannot move the median past the bulk of
values — so the median rejects impulses while the mean (box/Gaussian average) spreads them.

**S12.** Under axioms of linearity, shift/scale invariance, and *causality in scale* (no new extrema as `σ`
increases), the Gaussian is the unique generating kernel (Lindeberg, 1994; Koenderink, 1984). DoG =
`G_{kσ}−G_σ ≈ (k−1)σ²·∇²G_σ`, a scale-normalized Laplacian-of-Gaussian, whose extrema across scale are blob
detections — the SIFT keypoint mechanism (Lowe, 2004).
