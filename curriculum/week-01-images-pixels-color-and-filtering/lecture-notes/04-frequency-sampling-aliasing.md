# Lecture 4 — Frequency, the convolution theorem, and aliasing

So far filtering has lived in the spatial domain. Its true home is the **frequency domain**, and
moving there explains three things spatial reasoning cannot: *why* convolution is fast (FFT), *why* the
Gaussian is the right blur, and — most importantly for anyone who resizes images — *why aliasing happens and
how to prevent it*. This lecture is the signal-processing spine of computer vision (Oppenheim & Schafer,
*Discrete-Time Signal Processing*, 3rd ed., 2009; Gonzalez & Woods Ch. 4).

## Images as sums of sinusoids

The 2-D discrete Fourier transform writes an image as a sum of 2-D sinusoidal gratings:

    F[u, v] = Σ_m Σ_n f[m, n] exp(−j2π(um/H + vn/W)).

`F[u, v]` is the amplitude and phase of the grating with horizontal frequency `u` and vertical `v`. Low
frequencies (`u, v` near 0) carry smooth, large-scale content; high frequencies carry edges, fine texture,
and noise. Looking at `|F|` (the magnitude spectrum) is a diagnostic habit: sharp images have energy far from
the center; a blurred image's spectrum collapses toward the origin.

## The convolution theorem

The single most important identity in signal processing:

    f * h  ⇐⇒  F · H,

**convolution in space equals pointwise multiplication in frequency.** A filter is therefore fully described
by its **frequency response** `H`. This reframes every kernel: a blur is a **low-pass** filter (`H` keeps low
frequencies, kills high); a sharpen is a **high-boost**; a Sobel is a **band-pass** in one orientation. It
also gives fast convolution: for large kernels, `f * h = IFFT(FFT(f)·FFT(h))` costs `O(N log N)` instead of
`O(N k²)` — the reason libraries switch to FFT-based convolution past a kernel-size threshold. The box
filter's frequency response is a `sinc`, whose side-lobes are the ringing you see; the Gaussian's transform
is another Gaussian — strictly positive, no side-lobes — which is *why* it blurs cleanly.

## The sampling theorem

Return to Lecture 1: an image is a *sampled* signal, sampled at pixel pitch `Δ`, i.e. sampling frequency
`f_s = 1/Δ`. The **Nyquist–Shannon sampling theorem** (Shannon, 1949, "Communication in the presence of
noise," *Proc. IRE*) states: a signal is perfectly reconstructible from its samples if and only if it
contains no energy above `f_s/2` (the **Nyquist frequency**). Content above Nyquist cannot be represented —
it does not vanish, it **folds back** (aliases) onto lower frequencies, masquerading as content that was
never there.

## Aliasing: the bug behind moiré and jaggies

When a scene contains detail finer than the sampling grid can represent — a distant striped shirt, a picket
fence, a high-frequency texture — the aliased energy appears as **moiré** patterns and false low-frequency
structure. Downsampling is the worst offender: naive `img[::2, ::2]` (take every other pixel) doubles `Δ`,
halving Nyquist, so everything that was between the old and new Nyquist limits aliases. The fix is dictated by
the theorem: **low-pass filter first** (a Gaussian pre-blur) to remove energy above the new Nyquist, *then*
subsample. This is exactly what a correct `resize` does and what naive striding skips.

```python
# WRONG: aliases high-frequency texture into moiré
small = img[::2, ::2]

# RIGHT: band-limit to the new Nyquist, then subsample
blurred = gaussian_blur(img, sigma=1.0)     # remove energy above new Nyquist
small = blurred[::2, ::2]
```

Anti-aliasing is not optional polish; it is the sampling theorem enforced. The jagged staircase on a
diagonal line, the shimmering of a fence in a downscaled photo, the wagon-wheel effect in video — all one
phenomenon: spectral content above Nyquist folding down.

## Why this matters upstream of deep learning

CNNs stride and pool, which subsample feature maps — and Zhang (2019, "Making Convolutional Networks
Shift-Invariant Again," ICML) showed that stride-without-blur makes networks *not* shift-invariant, a direct
consequence of this lecture: an anti-aliasing blur before downsampling measurably improves accuracy and
robustness. The century-old sampling theorem is still correcting modern architectures.

**Takeaway:** the frequency domain is filtering's native habitat. The convolution theorem (`f*h ⇔ F·H`) makes
every filter a frequency response — blur = low-pass, sharpen = high-boost — and enables `O(N log N)` FFT
convolution. The sampling theorem sets a hard ceiling at Nyquist `f_s/2`; energy above it *aliases* into
moiré and jaggies, so any downsampling **must** be preceded by a low-pass pre-blur. This single fact governs
correct resizing, and even the striding of modern CNNs.
