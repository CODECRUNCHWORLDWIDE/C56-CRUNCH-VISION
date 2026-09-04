# Lecture 2 — Color spaces: from tristimulus to HSV and LAB

Color feels obvious until you have to compute with it, and then RGB reveals itself as only one
of infinitely many coordinate systems on a deeper object. The graduate version of this lecture starts not
with "RGB, HSV, LAB" as a list, but with the physical and perceptual theory those spaces are coordinates on:
**tristimulus colorimetry**, the CIE 1931 standard that makes color a measurable, transformable quantity
(Wyszecki & Stiles, *Color Science*, 2nd ed., 1982, is the reference).

## Color is a three-number projection of an infinite spectrum

Light entering the eye is a **spectral power distribution** `S(λ)` over wavelength — an infinite-dimensional
object. The human retina has three cone types (L, M, S) with response curves `l(λ), m(λ), s(λ)`. The eye
reports only three inner products:

    L = ∫ S(λ) l(λ) dλ,   M = ∫ S(λ) m(λ) dλ,   S = ∫ S(λ) s(λ) dλ.

This is **trichromacy**: color perception is a projection of an infinite-dimensional spectrum onto a 3-D
space. Its startling consequence is **metamerism** — physically different spectra with the same three
integrals look identical. That is *why* three numbers suffice for a display, and why "color" is a perceptual,
not purely physical, quantity. The CIE 1931 standard replaced the cone curves with standardized
color-matching functions `x̄, ȳ, z̄`, giving device-independent tristimulus values `(X, Y, Z)` — the ground
truth all other spaces reference.

## RGB: additive light, and a gamut

RGB models color as amounts of **red, green, and blue primaries added** together — matching how displays emit
and sensors capture, hence the default. But RGB is device-dependent: a set of primaries defines a *gamut*, a
triangle in the CIE chromaticity diagram, and no three real primaries span all visible colors. sRGB is the
standard small gamut; wide-gamut spaces (Adobe RGB, Display P3) reach further. Critically, RGB **entangles
color and brightness**: the same surface in shadow vs. sunlight has very different RGB triples though it is
"the same color," which makes RGB awkward for "find everything red regardless of lighting."

## Grayscale: luminance, done right

Many classical algorithms — edges, corners, most feature detectors — need only intensity. Converting RGB to
grayscale is a weighted sum matching human luminance sensitivity (we perceive green as brightest):

    Y = 0.299 R + 0.587 G + 0.114 B     (Rec. 601 luma)

The weights are not arbitrary — they approximate the luminance `Y` of CIE XYZ, and modern HD uses Rec. 709
weights (0.2126, 0.7152, 0.0722). A subtlety from Lecture 1: strictly correct luminance is computed on
*linear* RGB, then optionally re-encoded; the 0.299/0.587/0.114 formula is applied to gamma-encoded values by
convention (giving "luma," `Y'`, not true luminance). Grayscale is `(H, W)` — one-third the data and often all
the *structure* an algorithm needs.

## HSV: separating hue from brightness

**HSV** (hue, saturation, value) re-parameterizes RGB into a cylinder: **hue** (which color, an angle
0–360°), **saturation** (how vivid), **value** (how bright). It is a nonlinear but purely algebraic transform
of RGB — value is `max(R,G,B)`, saturation is the relative spread, hue is the angle of the dominant channel.
The win: a surface's *hue* stays roughly constant across illumination changes while brightness moves into
`V`. So "select all the orange pixels" is a clean band in `H` but a messy blob in RGB.

```python
import cv2
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))   # an orange band in hue
```

Caveat: hue is undefined for gray/near-black pixels (division by a vanishing saturation) and wraps at 360°,
so red straddles the seam — thresholding hue needs care at the boundary.

## LAB: perceptual uniformity

**CIE LAB** is a nonlinear transform of XYZ engineered so that **equal Euclidean distance ≈ equal perceived
color difference** — the property RGB badly lacks (a fixed RGB step is a large perceptual jump in some
regions, tiny in others). `L*` is lightness, `a*` green–red, `b*` blue–yellow. LAB is the space for
perceptual color-difference metrics (ΔE), for splitting an image into a lightness channel you can filter
independently, and for color-constancy work. Its cousin **YCbCr** separates luma from two chroma channels and
is the space JPEG and video compress in, because the eye tolerates far more chroma subsampling than luma
blur (4:2:0).

## The practical rule

Pick the space where your signal is *simplest to describe*. Color-based selection robust to lighting → HSV.
Structure/edges → grayscale/luma. Perceptual color difference or lightness-only filtering → LAB. Feeding a
neural net → usually normalized RGB, because the network learns whatever channel mixing it needs. The skill
is not memorizing conversion matrices but asking: *in which coordinate system is my target a simple region?*

**Takeaway:** color is a 3-D projection (trichromacy) of an infinite spectrum, formalized by CIE tristimulus
values; every color space is a coordinate change on that. RGB is device-dependent and entangles color with
brightness; grayscale/luma keeps structure; HSV isolates hue for lighting-robust selection; LAB makes
Euclidean distance perceptually meaningful. Compute in the space where your target is simplest.
