# Week 1 — Quiz

Fifteen questions spanning image formation and the sensor pipeline, tristimulus colorimetry and color spaces, LSI theory and convolution, the sampling theorem and aliasing, and nonlinear/edge-preserving filtering. Attempt each before the answer key.

**1. A digital image differs from the continuous scene it captures because it undergoes:**

- A. only amplitude quantization, never spatial sampling
- B. a Fourier transform that discards phase
- C. spatial sampling and amplitude quantization
- D. a lossless one-to-one mapping

<details>
<summary>Answer</summary>

**C. spatial sampling and amplitude quantization** — A digital image is the continuous irradiance field sampled on a spatial lattice and quantized in intensity — two independent discretizations.

</details>

**2. Stored 8-bit sRGB pixel values are gamma-encoded, which means that averaging raw pixel values:**

- A. is impossible without a GPU
- B. changes the image resolution
- C. correctly averages the physical light
- D. does NOT average physical light — you must linearize first

<details>
<summary>Answer</summary>

**D. does NOT average physical light — you must linearize first** — sRGB applies ~V^(1/2.2); raw values are perceptual, not linear in light, so physically correct averaging/blurring must linearize, operate, then re-encode.

</details>

**3. A '12-megapixel color photo' from a Bayer-sensor camera actually measured:**

- A. 12M independent samples of each of R, G, B
- B. only luminance, with color added in software randomly
- C. one color per photosite (mosaic), with the other two channels interpolated (demosaiced)
- D. 36M raw color samples

<details>
<summary>Answer</summary>

**C. one color per photosite (mosaic), with the other two channels interpolated (demosaiced)** — A Bayer color-filter array measures one color per photosite (50% G, 25% R, 25% B); the full RGB image is interpolated, so two-thirds of each channel is inferred.

</details>

**4. Human color perception is trichromatic, meaning a color is fundamentally:**

- A. a single wavelength
- B. independent of the illuminant
- C. always exactly reproducible by any three primaries
- D. a 3-D projection (three cone integrals) of an infinite-dimensional spectrum

<details>
<summary>Answer</summary>

**D. a 3-D projection (three cone integrals) of an infinite-dimensional spectrum** — Three cone types report three inner products of the spectrum, projecting an infinite-dimensional signal to 3-D — which is why metamers exist and three numbers suffice for a display.

</details>

**5. Two physically different spectra that look identical to a human are called:**

- A. metamers
- B. aliases
- C. harmonics
- D. chromatic aberrations

<details>
<summary>Answer</summary>

**A. metamers** — Metamerism is the direct consequence of trichromacy: different spectra with the same three cone integrals are perceptually identical.

</details>

**6. HSV is preferred over RGB for lighting-robust color selection because:**

- A. hue stays roughly constant under illumination changes while brightness moves into V
- B. it is a linear transform of XYZ
- C. it uses less memory
- D. it removes all color information

<details>
<summary>Answer</summary>

**A. hue stays roughly constant under illumination changes while brightness moves into V** — HSV separates which-color (hue) from how-bright (value), so a target color is a clean band in H even as lighting shifts brightness into V.

</details>

**7. CIE LAB is engineered so that:**

- A. the three channels are the raw sensor primaries
- B. hue is undefined everywhere
- C. it is the fastest space to convolve in
- D. equal Euclidean distance approximates equal perceived color difference

<details>
<summary>Answer</summary>

**D. equal Euclidean distance approximates equal perceived color difference** — LAB is perceptually (near-)uniform: a fixed numeric distance corresponds to a roughly constant perceived difference — the property RGB lacks.

</details>

**8. Any operator that is both linear and shift-invariant can be written as:**

- A. a matrix inverse
- B. a nonlinear diffusion PDE
- C. a convolution with a fixed kernel (its impulse response)
- D. a median filter

<details>
<summary>Answer</summary>

**C. a convolution with a fixed kernel (its impulse response)** — This is the defining theorem of LSI systems: linearity + shift-invariance forces the operator to be convolution with the impulse response.

</details>

**9. What most libraries and every CNN call 'convolution' is technically:**

- A. a Fourier transform
- B. matrix inversion
- C. cross-correlation (no flip)
- D. true convolution with a kernel flip

<details>
<summary>Answer</summary>

**C. cross-correlation (no flip)** — Libraries skip the flip; for symmetric kernels it is identical and for learned kernels the network just learns the flipped weights, so 'convolution' means correlation.

</details>

**10. A Gaussian blur is separable, so a k×k Gaussian costs, per pixel:**

- A. 2k multiplies via two 1-D passes
- B. k! multiplies
- C. k² multiplies (unavoidable)
- D. one multiply

<details>
<summary>Answer</summary>

**A. 2k multiplies via two 1-D passes** — exp(-(i²+j²)/2σ²) factors into an outer product, so f*h = (f*a)*b — two 1-D passes, 2k multiplies instead of k².

</details>

**11. The convolution theorem states that convolution in the spatial domain corresponds to:**

- A. a matrix inverse in the frequency domain
- B. nothing — the domains are unrelated
- C. pointwise multiplication in the frequency domain
- D. addition in the frequency domain

<details>
<summary>Answer</summary>

**C. pointwise multiplication in the frequency domain** — f*h ⇔ F·H: a spatial convolution is a pointwise product of the two Fourier transforms, which is why filters are frequency responses and FFT convolution is O(N log N).

</details>

**12. The Nyquist–Shannon sampling theorem says a signal sampled at rate f_s is recoverable only if it contains no energy above:**

- A. f_s/2 (the Nyquist frequency)
- B. 2·f_s
- C. f_s
- D. zero frequency

<details>
<summary>Answer</summary>

**A. f_s/2 (the Nyquist frequency)** — Content above half the sampling frequency cannot be represented; it folds back (aliases) onto lower frequencies.

</details>

**13. Downsampling an image with plain striding img[::2, ::2] and no pre-blur causes:**

- A. aliasing — high-frequency detail folds into false moiré patterns
- B. gamma correction
- C. an increase in resolution
- D. a lossless half-size copy

<details>
<summary>Answer</summary>

**A. aliasing — high-frequency detail folds into false moiré patterns** — Striding halves the sampling rate; energy between old and new Nyquist aliases, producing moiré and jaggies. A low-pass pre-blur is mandatory.

</details>

**14. A median filter beats a Gaussian blur on salt-and-pepper noise because:**

- A. the median is an order statistic unaffected by a few extreme outliers, and it preserves steps
- B. it runs in the frequency domain
- C. it is a linear shift-invariant filter
- D. it averages the outliers into the result

<details>
<summary>Answer</summary>

**A. the median is an order statistic unaffected by a few extreme outliers, and it preserves steps** — As an order statistic the median ignores a minority of extreme values (impulses) and keeps edges sharp — a convolution/average merely spreads the impulses into gray smudges.

</details>

**15. The bilateral filter preserves edges while smoothing because it weights each neighbor by:**

- A. spatial distance only, like a Gaussian
- B. its frequency alone
- C. spatial distance AND intensity (range) difference, so neighbors across an edge get near-zero weight
- D. a fixed learned kernel

<details>
<summary>Answer</summary>

**C. spatial distance AND intensity (range) difference, so neighbors across an edge get near-zero weight** — The extra range Gaussian on |I_p - I_q| kills the contribution of neighbors across a strong edge, so the filter averages within a region but not across its boundary — making it nonlinear and non-convolutional.

</details>

---
