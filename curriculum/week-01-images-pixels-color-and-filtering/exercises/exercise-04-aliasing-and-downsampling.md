# Exercise 4 — See aliasing and defeat it with the sampling theorem

**Goal:** turn Lecture 4 from theory into something you have watched happen and fixed.

## Tasks

1. Find or synthesize a high-frequency image: a photo of a striped shirt/fence, or a synthetic radial
   chirp/zone-plate pattern (`sin(k·(x²+y²))`) whose frequency rises toward the edges.
2. **Naive downsample** by 2× and 4× with plain striding `img[::2, ::2]`. Display the result and point to the
   moiré patterns and false low-frequency structure. On the zone plate, the aliasing is unmistakable.
3. **Correct downsample:** Gaussian pre-blur (choose σ ≈ new pixel pitch) *then* stride. Display and confirm
   the moiré is gone. Explain, in one paragraph, which spectral energy the pre-blur removed and why the
   theorem requires it.
4. Compute and display the 2-D FFT magnitude spectrum (`np.fft.fft2`, `fftshift`, log scale) of the original,
   the naive-downsampled, and the pre-blurred-downsampled images. Point to where high-frequency energy folded
   back in the naive case.
5. Build a 3-level Gaussian pyramid (correct anti-aliased downsampling at each level) and display it.

## Deliverable

A notebook contrasting naive vs. anti-aliased downsampling on a high-frequency image, the three FFT spectra
with the aliasing annotated, and a Gaussian pyramid — plus a short paragraph stating the Nyquist frequency at
each downsampling step and why pre-blur is mandatory, not optional.
