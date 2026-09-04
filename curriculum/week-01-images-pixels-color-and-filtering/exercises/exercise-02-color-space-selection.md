# Exercise 2 — Select an object by color across color spaces

**Goal:** experience why color space matters by doing a task that is easy in one space and hard
in another, and connect it to tristimulus/colorimetric reasoning.

## Tasks

1. Find an image with a distinctly colored object (red ball, blue sign). Try to select just that object with
   thresholds on RGB channels. Observe how lighting variation makes it fragile, and explain *why* in terms of
   RGB entangling color with brightness.
2. Convert to HSV. Select the object with a band on the **hue** channel plus loose saturation/value bounds
   (`cv2.inRange`). Handle the red hue-wrap-around at 0/180 explicitly (two bands OR'd together).
3. Convert to **LAB** and select using the `a*`/`b*` chroma channels. Compare the LAB selection to the HSV
   one; comment on which gives a cleaner boundary and why LAB's perceptual uniformity helps.
4. Change lighting (a shadowed and a bright example, or dim the image) and show which selection survives.
   Quantify with IoU against a hand-drawn mask if you can.

## Deliverable

A notebook with the RGB, HSV, and LAB masks side by side, an IoU or qualitative comparison under two lighting
conditions, and three sentences explaining why hue/chroma-based selection is more robust to lighting than RGB
thresholds — grounded in the color-vs-brightness entanglement of RGB.
