# Week 1 — Reading List

Primary sources for Week 1. Start with the three starred essentials — a modern CV text, the colorimetry reference, and the sampling paper — then read for depth. Prefer these to blog posts; they are where the ideas are stated correctly.

- **Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., 2022), Ch. 2–3** — image formation, the sensor pipeline, and image processing/filtering; the canonical graduate reference for this whole week. Free PDF online.
- **Wyszecki & Stiles, *Color Science: Concepts and Methods, Quantitative Data and Formulae* (2nd ed., 1982)** — the standard reference for CIE tristimulus colorimetry, metamerism, and color spaces underlying Lecture 2.
- **Shannon (1949), 'Communication in the Presence of Noise,' *Proc. IRE*** — the sampling theorem itself; the origin of Nyquist, aliasing, and why you must pre-blur before downsampling (Lecture 4).
- Gonzalez & Woods, *Digital Image Processing* (4th ed., 2018), Ch. 3–4 — spatial and frequency-domain filtering and the convolution theorem, worked in careful detail.
- Tomasi & Manduchi (1998), 'Bilateral Filtering for Gray and Color Images,' ICCV — the edge-preserving filter of Lecture 5, stated by its inventors.
- Perona & Malik (1990), 'Scale-Space and Edge Detection Using Anisotropic Diffusion,' IEEE TPAMI — filtering as an edge-stopping diffusion PDE; the deep view of smoothing.
- Lindeberg (1994), *Scale-Space Theory in Computer Vision* (or the 1994 J. Applied Statistics article) — why the Gaussian uniquely generates a scale space; the bridge to Week 2 features.
- Burt & Adelson (1983), 'The Laplacian Pyramid as a Compact Image Code,' IEEE Trans. Communications — the (anti-aliased) image pyramid, foundational for multi-scale vision.
- Wang, Bovik, Sheikh & Simoncelli (2004), 'Image Quality Assessment: From Error Visibility to Structural Similarity,' IEEE TIP — SSIM, the perceptual metric you should use over PSNR in the denoising challenge.
- Zhang (2019), 'Making Convolutional Networks Shift-Invariant Again,' ICML — modern proof that CNN strides alias; the sampling theorem correcting deep architectures.
- Poynton, *Digital Video and HD: Algorithms and Interfaces* (2nd ed., 2012), gamma/luma chapters — the authoritative account of gamma encoding, luma vs. luminance, and chroma subsampling.
