# Exercise 1 — Build, tune, and justify a Canny edge detector

**Goal:** understand Canny by driving every knob *and* connecting each to Canny's optimality criteria.

## Tasks

1. Load a grayscale image and run `cv2.Canny` with several threshold pairs (e.g. low/high of 50/150,
   100/200, 150/250). Display each result side by side and describe how the strong/weak/off partition
   shifts.
2. **The blur / detection trade-off.** Run Canny after Gaussian blurs of σ ∈ {0.5, 1.4, 3.0}. Show how
   small σ keeps fine detail but admits noise, while large σ suppresses noise but loses and mislocates
   edges. Tie this to Canny's detection-vs-localization tension.
3. **Build the front end yourself.** From your Week-1 Sobel filter, implement the first stages by hand:
   gradient magnitude and orientation, then threshold the magnitude. Compare your crude edges to
   OpenCV's Canny and describe in writing what non-maximum suppression and hysteresis each add.
4. **Implement NMS.** Add non-maximum suppression to your hand-built detector: quantize the gradient
   orientation to the nearest of {0, 45, 90, 135}° and keep a pixel only if its magnitude exceeds both
   neighbours along the gradient. Show the before/after thinning.
5. **Auto-Canny.** Set the thresholds automatically from the image median
   (`low = 0.66·median`, `high = 1.33·median`) and test on three images with different lighting.

## Deliverable

A notebook comparing threshold and σ settings, your hand-built gradient+NMS edges vs. full Canny, and the
auto-threshold results, with two–three sentences mapping NMS to the single-response criterion and
hysteresis to the detection/localization trade-off. You should be able to explain every stage from the
theory, not just the API.
