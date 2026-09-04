# Challenge 1 — Design a kernel from its frequency response

You now know (Lectures 3–4) that a kernel *is* an impulse response and that its effect is best
understood as a frequency response. Design kernels for specific goals and defend each choice from both the
spatial and the frequency viewpoint.

1. Design a kernel that **emphasizes vertical edges** but ignores horizontal ones, and one that does the
   reverse. Show both on an image with clear vertical and horizontal structure (a building, a fence), and
   sketch/compute each kernel's 2-D frequency response to argue it is an oriented band-pass.
2. Design an **emboss** filter (a directional kernel producing a pressed-metal look) and explain
   geometrically *and* spectrally why it produces that 3-D illusion.
3. Compare a **box** blur and a **Gaussian** blur of similar spatial extent on a noisy image. Which leaves
   fewer artifacts? Show each kernel's frequency response (the box's sinc side-lobes vs. the Gaussian's clean
   roll-off) and connect the ringing you see to those side-lobes.
4. Build an **unsharp mask** (`sharpened = original + amount·(original − blurred)`) and relate it to the
   `[[0,−1,0],[−1,5,−1],[0,−1,0]]` sharpen kernel — show they are the same idea.

**Deliverable:** a note with each kernel, its output, its (computed or sketched) frequency response, and a
paragraph reasoning from the numbers *and* the spectrum to the visual result. The graded skill is designing a
filter by what it does to frequencies, not copying kernels from a table.
