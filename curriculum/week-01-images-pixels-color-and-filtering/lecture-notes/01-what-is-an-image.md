# Lecture 1 — An image is a sampled, quantized radiance field

A photograph is a picture to you and a **sampled, quantized signal** to a computer. The
naive statement — "an image is a grid of numbers" — is correct but hides the three physical steps that
produced those numbers: **formation** (a scene's radiance projected through a lens onto a sensor plane),
**sampling** (the continuous irradiance read out at a discrete grid of photosites), and **quantization**
(each sample rounded to a finite-precision integer). Understanding these three steps is what separates a
practitioner who debugs vision systems from one who guesses. Szeliski (*Computer Vision: Algorithms and
Applications*, 2nd ed., 2022, §2.3) is the canonical reference for this pipeline.

## The continuous signal behind the grid

Model the image as a continuous function `f(x, y)` giving irradiance at every point of the sensor plane.
A digital image is `f` **sampled** on a lattice and **quantized** in amplitude:

    I[m, n] = Q( f(mΔx, nΔy) ),   m = 0..H−1, n = 0..W−1,

where `Δx, Δy` are the pixel pitch and `Q(·)` maps a real intensity to, most commonly, an 8-bit integer in
`[0, 255]`. Two independent discretizations are happening: **spatial** (how densely we sample → resolution,
and, as Lecture 4 shows, whether we alias) and **tonal** (how finely we quantize → bit depth, banding).
Confusing them is a classic error: an 8-megapixel image quantized to 4 bits looks sharp but posterized.

## Value range, dtype, and the gamma trap

In the most common encoding each pixel is an unsigned 8-bit integer, `0` black to `255` white. Many
libraries instead use `float32` in `[0, 1]`. Displaying a float image as if it were uint8 (or vice versa)
is the single most common beginner bug — a float image shown on a 0–255 scale looks all black.

```python
import numpy as np
from PIL import Image
img = np.asarray(Image.open("cat.jpg").convert("L"))   # L = grayscale
print(img.shape, img.dtype, img.min(), img.max())       # (H, W) uint8 0 255
```

But there is a subtler trap: the stored 8-bit values are usually **gamma-encoded**, not linear in light.
Display and file formats (sRGB) apply roughly `V_encoded ≈ L_linear^(1/2.2)` so that the limited code
values are perceptually uniform — human vision is far more sensitive to relative differences in dark tones
(a consequence of Weber's law). This means **averaging raw sRGB pixels is not averaging light**. Physically
correct blurring, resizing, and alpha compositing must first *linearize* (undo gamma), operate, then
re-encode. Skipping this is why naive downscaling darkens bright thin features. Poynton's *Digital Video
and HD* (2nd ed., 2012) is the standard treatment.

## Color adds a channel axis — and a sensor story

A color image is 3-D, `(H, W, C)` with `C = 3` for red, green, blue. A pixel is a length-3 vector. A batch
for a network is 4-D: `(N, C, H, W)` in PyTorch convention — channels *before* spatial dims, the opposite
of Pillow/OpenCV's `(H, W, C)`. Half of all vision bugs are a channel axis in the wrong place; print shapes
relentlessly.

Where do three channels come from? Most sensors are **monochrome** photodiodes with a **color filter array**
(the Bayer mosaic: 50% green, 25% red, 25% blue, per Bayer's 1976 Kodak patent) glued on top. Each photosite
measures only one color; the full RGB image is **interpolated** (demosaiced). So a "12-megapixel color
photo" never truly measured 12M red, 12M green, and 12M blue samples — two-thirds of every channel is
inferred. This is why raw files, JPEG artifacts, and demosaicing halos exist, and why chroma is lower
resolution than luma in practice.

## Channel order and layout gotchas

- **Pillow** returns RGB; **OpenCV** returns **BGR** — a historical quirk. Feed Pillow-RGB into OpenCV code
  expecting BGR and your reds and blues swap silently.
- **PyTorch/torchvision** want `(C, H, W)`, float, usually normalized; Pillow gives `(H, W, C)`, uint8.
  `torchvision.transforms` bridges them.
- Memory layout matters too: an image is a strided view over a flat buffer; a transpose or channel-split can
  be a metadata trick (a view) or a copy — knowing which controls performance.

## Resolution, resizing, and lost information

More pixels means more detail and more compute. Networks want a fixed input size, so you resize constantly —
and downsizing **destroys information you can never recover**, while any resize changes aspect ratio unless
you letterbox or crop. Crucially, downsampling *must* be preceded by a low-pass (anti-alias) filter or you
get aliasing (Lecture 4); "just take every other pixel" is a bug, not an optimization. Keep the original.

## Worked example: what the histogram tells you

Plot the intensity histogram of a grayscale image. A **dark** image piles counts near 0; a **bright** one
near 255; a **low-contrast** one bunches in a narrow band. Histogram *equalization* remaps intensities so the
cumulative distribution is roughly linear, spreading contrast — a one-line operation whose justification is
pure probability (mapping a variable through its own CDF yields a uniform distribution). You never trust a
vision number you have not seen; the histogram is the first thing you look at.

**Takeaway:** an image is the sampling (spatial) and quantization (tonal) of a continuous radiance field —
not just a matrix. Know your shape and layout (`(H,W)`, `(H,W,3)`, `(N,C,H,W)`), your dtype and range
(`uint8` 0–255 vs `float` 0–1), your channel order (RGB/BGR/CHW), and — the graduate detail — that stored
values are usually *gamma-encoded* and came through a *Bayer/demosaic* pipeline. Print shapes; look at
histograms; linearize before you average light.
