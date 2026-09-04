# Week 1 — Images, pixels, color & filtering

> **Goal:** by Sunday you can (1) explain an image as the sampling and quantization of a continuous radiance field, and reason precisely about shape, dtype, channel order, gamma, and the sensor model that produced the numbers; (2) treat color not as 'RGB' but as tristimulus coordinates in a colorimetric space, and convert between RGB, grayscale, HSV, and CIE LAB with the physics and the matrices in hand; and (3) derive convolution as the impulse response of a linear shift-invariant system, state and use the convolution theorem, predict aliasing from the Nyquist limit, and implement blur/sharpen/gradient/edge-preserving filters from scratch — no library hiding the operation.

Welcome to **C56 · Crunch Vision**. This is the graduate on-ramp, and it earns its keep before a single neural network appears. Everything the next eleven weeks do — features, CNNs, detection, segmentation, transformers over patches — is arithmetic on a grid of numbers, and if that grid is a black box the bugs never leave. So this week we open the box completely.

The usual first week says 'an image is a matrix of pixels; convolution slides a kernel.' True, and useless if you stop there. We go under it. An image is not fundamentally a matrix — it is the **sampling and quantization of a continuous two-dimensional signal**, the irradiance falling on a sensor, itself the projection of a scene's radiance through a lens. That single reframing (Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., 2022, Ch. 2) explains gamma, dynamic range, demosaicing, aliasing, and why resizing is dangerous. Color, likewise, is not 'three numbers' — it is a projection of an infinite-dimensional spectrum onto three cone responses, formalized by the CIE 1931 tristimulus theory, and every color space is a coordinate change on that projection. And filtering is not 'a trick with kernels' — it is the theory of **linear shift-invariant systems**, where convolution is the impulse response and the Fourier transform diagonalizes everything.

You will implement convolution by hand, derive the Gaussian's separability, meet the sampling theorem and see aliasing with your own eyes, and build a bilateral filter that a box blur cannot match. When torchvision does all this for you in Week 3, it will be a faster version of machinery you already own — never magic.

## Learning objectives

By the end of this week, you will be able to:

- **Model** an image as the sampling and quantization of a continuous irradiance field, and reason about shape, dtype, value range, channel order, gamma encoding, and the raw-sensor (Bayer/demosaic) pipeline that produced it.
- **Convert** between color representations — RGB, luminance grayscale, HSV, and CIE LAB — grounded in tristimulus colorimetry, and choose the space in which a given target is a simple region.
- **Derive** 2-D convolution as the impulse response of a linear shift-invariant system, distinguish it from cross-correlation, and implement it from scratch with correct padding and stride.
- **Prove** and exploit separability of the Gaussian kernel, reducing a k×k filter to two 1-D passes, and state the convolution theorem relating spatial convolution to frequency-domain multiplication.
- **Predict** aliasing from the Nyquist–Shannon sampling theorem, explain why downsampling requires a pre-blur, and diagnose moiré and jaggies as spectral overlap.
- **Implement** edge-preserving and scale-space filters — median, bilateral, and a Gaussian pyramid — and explain where linear convolution fails and a nonlinear operator wins.
- **Compute** gradient magnitude and orientation with Sobel/derivative-of-Gaussian filters, previewing Week 2's edge and feature detectors.
- **Diagnose** the common failure modes — swapped channels, dtype/range confusion, gamma mistakes, and aliasing — from the pixels and histograms alone, never trusting a number you have not seen.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CAP 4410` — explain image formation, treat a digital image as the sampling and quantization of a continuous signal, and apply linear filtering and convolution to it. |
| Industry | Verify that an imaging pipeline is doing what you believe it is — channel order, dtype, value range, gamma, and whether a resize threw detail away — before any model is blamed for the output. |
| Beyond the bar | derives convolution as the impulse response of a linear shift-invariant system and requires aliasing to be shown in the FFT magnitude spectrum rather than asserted — `mini-project/README.md` |

## Prerequisites

- Python and NumPy: array creation, slicing, broadcasting, vectorized ops.
- Single- and multivariable calculus: partial derivatives, integrals, the notion of a convolution integral.
- Linear algebra: matrix multiplication, eigen/singular values conceptually, change of basis.
- Basic signals/Fourier at an intuitive level helps but is not assumed — Lecture 4 builds what you need.

## This week

**Lectures**

1. [Lecture 1 — An image is a sampled, quantized radiance field](lecture-notes/01-what-is-an-image.md)
2. [Lecture 2 — Color spaces: from tristimulus to HSV and LAB](lecture-notes/02-color-spaces.md)
3. [Lecture 3 — Convolution: the impulse response of a linear shift-invariant system](lecture-notes/03-convolution-and-filtering.md)
4. [Lecture 4 — Frequency, the convolution theorem, and aliasing](lecture-notes/04-frequency-sampling-aliasing.md)
5. [Lecture 5 — Beyond linear filters: edge-preserving filtering and scale space](lecture-notes/05-nonlinear-and-scale-space.md)

**Exercises**

1. [Exercise 1 — Load, inspect, and interrogate the sensor pipeline](exercises/exercise-01-load-and-inspect.md)
2. [Exercise 2 — Select an object by color across color spaces](exercises/exercise-02-color-space-selection.md)
3. [Exercise 3 — Implement convolution, verify separability, and cross-check](exercises/exercise-03-convolution-by-hand.md)
4. [Exercise 4 — See aliasing and defeat it with the sampling theorem](exercises/exercise-04-aliasing-and-downsampling.md)

**Challenges**

1. [Challenge 1 — Design a kernel from its frequency response](challenges/challenge-01-kernel-design.md)
2. [Challenge 2 — Make convolution fast: separability and the FFT](challenges/challenge-02-separable-and-speed.md)
3. [Challenge 3 — Denoise without destroying edges](challenges/challenge-03-edge-preserving-denoise.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 2.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A from-scratch notebook (no cv2.filter2D or scipy for the core convolution) that: loads an image and reports shape/dtype/range/gamma; converts among RGB, luminance grayscale, HSV, and CIE LAB and shows a histogram per channel; implements convolve2d with padding and stride, verified against a library; builds blur (box, Gaussian — proven separable), sharpen, and Sobel gradient-magnitude filters; demonstrates aliasing by naive vs. pre-blurred 2× downsampling; and adds one edge-preserving filter (median or bilateral) shown to beat Gaussian blur on salt-and-pepper or edge-preservation. A short writeup ties each result to the theory.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
