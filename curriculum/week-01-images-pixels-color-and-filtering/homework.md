# Week 1 — Homework

Consolidate Week 1 and turn it into instinct. These take a few focused hours and set up Week 2's move to edges, features, and descriptors — all of which are filtering plus scale space. Do the derivations and the aliasing demo by hand before leaning on a library; the pixel-level intuition must be muscle memory before torchvision hides it.

## Tasks

- Write, in your own words, the three physical steps (formation, sampling, quantization) that turn a scene into a digital image, and give one bug each mis-step causes (gamma darkening, aliasing, banding).
- Take one image and produce a labeled figure of it in RGB channels, luminance grayscale, HSV channels, and LAB channels side by side; add one sentence per space on when you would compute in it.
- Prove on paper that a Gaussian kernel is separable and state the per-pixel multiply count for a k×k Gaussian done directly vs. as two 1-D passes.
- Demonstrate aliasing: downsample a high-frequency image (fence or synthetic zone plate) 2× with naive striding vs. Gaussian-pre-blur-then-stride, and explain via the sampling theorem which spectral energy folded.
- Extend your convolution to accept a stride argument and confirm stride 2 halves the output dimensions using floor((n+2p−k)/s)+1; note that stride-without-pre-blur aliases feature maps (Zhang 2019).
- Implement a median OR bilateral filter from scratch and show, with a PSNR or SSIM number, that it beats a Gaussian blur on either salt-and-pepper noise (median) or edge preservation under Gaussian noise (bilateral).

## Definition of done

A committed notebook/module implementing convolution from scratch (with padding and stride, verified against a library), a filter bank (box, separable Gaussian with the 2-D equality shown, sharpen/unsharp, Sobel magnitude+orientation), correct RGB/grayscale/HSV/LAB conversions with a gamma linear-vs-encoded blur demo, an anti-aliased 2× downsampler contrasted with naive striding plus FFT spectra and a Gaussian pyramid, and at least one nonlinear (median or bilateral) filter beating Gaussian blur with a PSNR/SSIM number — all with no reliance on cv2.filter2D for the core convolution, and a writeup tying every result to the theory.

Submit by committing your work to your course repo under `week-01/`.
