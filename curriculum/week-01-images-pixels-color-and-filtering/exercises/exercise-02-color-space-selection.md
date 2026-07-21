# Exercise 2 — Select an object by color in HSV

**Goal:** experience why color space matters by doing a task that is easy in one space and hard in another.

## Tasks

1. Find an image with a distinctly colored object (a red ball, a blue sign). Try to select just that object with a threshold on RGB channels. Observe how lighting variation makes it fragile.
2. Convert the image to HSV. Select the object with a band on the **hue** channel plus loose saturation/value bounds (`cv2.inRange`).
3. Display the resulting mask overlaid on the image. Compare the RGB-threshold result to the HSV-hue result.
4. Change the object's lighting (find a shadowed and a bright example, or dim the image) and show which selection survives.

## Deliverable

A notebook with both masks side by side and two sentences explaining why hue-based selection is more robust to lighting than RGB thresholds. This is the intuition behind choosing a color space.
