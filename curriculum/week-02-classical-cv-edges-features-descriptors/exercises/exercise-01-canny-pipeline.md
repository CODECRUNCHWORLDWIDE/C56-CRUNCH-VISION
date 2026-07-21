# Exercise 1 — Build and tune a Canny edge detector

**Goal:** understand Canny by driving every knob.

## Tasks

1. Load a grayscale image and run `cv2.Canny` with several threshold pairs (e.g. low/high of 50/150, 100/200, 150/250). Display each result.
2. Show the effect of the initial Gaussian blur: run Canny on the raw image vs. a blurred image and compare the noise in the edge maps.
3. Implement, from your Week-1 Sobel filter, the first two Canny stages yourself (gradient magnitude and direction) and threshold the magnitude. Compare your crude edges to OpenCV's Canny and describe what non-max suppression and hysteresis add.
4. Pick threshold values automatically from the image median and test on three different images.

## Deliverable

A notebook comparing threshold settings and your hand-built gradient edges vs. full Canny, with two sentences on what NMS and hysteresis contribute. You should be able to explain every stage.
