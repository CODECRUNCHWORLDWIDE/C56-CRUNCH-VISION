# Mini-Project — A Hand-Built Image Filter Toolkit

## Brief

Build a small image-processing toolkit *from scratch* — your own convolution plus a set of filters — proving you understand pixels, color, and the operation that powers the rest of the course before any network does it for you.

## Requirements

1. **Loading & color:** load an image, and provide functions to convert between RGB, grayscale (hand-written weighted sum), and HSV. Show a histogram.
2. **Convolution:** your own `convolve2d` with padding and stride, verified against a library on at least one kernel.
3. **Filters:** at least a box blur, a Gaussian blur, a sharpen, and Sobel x/y gradients with a gradient-magnitude edge output.
4. **Gallery:** a labeled grid showing every filter applied to at least two real images — original next to result.
5. **README/notes:** for each filter, one line on what the kernel does and why.

## Stretch

- Add a median filter (not a convolution — a nonlinear neighborhood operation) and show it beats Gaussian blur at removing salt-and-pepper noise.
- Implement Gaussian blur as separable 1-D passes and confirm the speedup.

## What you're proving

You can treat an image as data, move between color spaces on purpose, and implement convolution — the single operation that connects classical filtering to deep CNNs. Everything from here builds on this grid of numbers, and you now own it.
