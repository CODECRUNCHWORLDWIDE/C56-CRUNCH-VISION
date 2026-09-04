# Mini-Project — A Hand-Built Imaging & Filtering Toolkit

## Brief

Build a small image-processing toolkit *from scratch* — your own convolution, a filter bank, correct color
conversions, and a correct (anti-aliased) resizer — proving you understand the image as a sampled, quantized,
gamma-encoded, color-projected signal, and the LSI operation that powers the rest of the course, before any
network does it for you.

## Requirements

1. **Loading, gamma & color.** Load an image; report shape/dtype/range. Provide conversions among RGB,
   luminance grayscale (hand-written weighted sum), HSV, and CIE LAB. Show a per-channel histogram. Include a
   `linearize`/`encode` gamma pair and demonstrate the difference between blurring in gamma vs. linear space
   on a bright thin feature.
2. **Convolution.** Your own `convolve2d(img, kernel, stride)` with edge and zero padding, verified against a
   library on at least one kernel (mind correlation vs. convolution).
3. **Filter bank.** At least: box blur, Gaussian blur (implemented as **separable** 1-D passes, with the
   full-2-D equality demonstrated), sharpen/unsharp mask, and Sobel x/y with gradient-magnitude and
   orientation outputs.
4. **Anti-aliased resize.** A `downsample_2x` that Gaussian-pre-blurs then subsamples, contrasted against
   naive striding on a high-frequency (fence/zone-plate) image, with the FFT magnitude spectra showing the
   aliasing folded energy. Build a 3-level Gaussian pyramid.
5. **One nonlinear filter.** Add a median *or* a from-scratch bilateral filter, and show it beats Gaussian
   blur on salt-and-pepper noise (median) or on edge preservation under Gaussian noise (bilateral), reported
   with PSNR/SSIM and edge crops.
6. **Gallery + notes.** A labeled grid of every filter on at least two real images (original next to result),
   and for each filter one line on what the kernel/operator does and why, tied to the lecture theory.

## Stretch

- Implement FFT-based convolution (`IFFT(FFT(f)·FFT(h))` with zero-padding) and find the kernel size where it
  overtakes direct convolution; explain the `O(Nk²)` vs `O(N log N)` crossover.
- Add anisotropic (Perona–Malik) diffusion and compare its edge preservation to the bilateral filter.

## What you are proving

You can treat an image as a physically produced, sampled signal; move between color spaces on purpose and for
reasons; implement convolution as an LSI operation and reason about its speed via separability and the FFT;
respect the sampling theorem when you resize; and reach for a nonlinear filter when linearity fails.
Everything from here — CNNs, detection, transformers over patches — is a faster, learned version of exactly
this machinery, and you now own it.
