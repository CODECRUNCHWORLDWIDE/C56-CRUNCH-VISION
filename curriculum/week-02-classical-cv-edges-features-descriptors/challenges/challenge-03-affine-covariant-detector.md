# Challenge 3 — Push invariance: affine-covariant regions and viewpoint change

SIFT/ORB are (approximately) similarity-invariant — scale + rotation + translation. Real viewpoint
change is **affine** (and, at extremes, projective): as you circle an object, local patches shear. This
open challenge investigates how far invariance can be pushed with classical machinery.

1. **Read and summarize** the idea of *affine-covariant* regions: Harris-Affine / Hessian-Affine
   (Mikolajczyk & Schmid, 2004, "Scale & affine invariant interest point detectors," *IJCV*) and MSER
   (Matas et al., 2002, "Robust wide-baseline stereo from maximally stable extremal regions," *BMVC*).
   Explain what "covariant" means: the region deforms *with* the image so the normalized patch is stable.
2. **Experiment.** Take an image and apply increasing out-of-plane rotation (simulate with strong affine
   warps, or photograph a poster from increasing angles). Measure the repeatability and match count of ORB
   / SIFT as the viewpoint angle grows. Where does similarity-invariance break?
3. **Compare** against `cv2` MSER regions (or an affine-adapted detector if available) on the same
   sequence. Does affine covariance extend the usable viewpoint range? Quantify.
4. **Analyze.** At what point does *no* classical detector suffice, and a detector-free / learned dense
   matcher becomes necessary? Relate to the invariance hierarchy: translation ⊂ similarity ⊂ affine ⊂
   projective.

**Deliverable:** a short report with a repeatability-vs-viewpoint-angle plot for at least two detectors, a
clear statement of where each invariance class breaks, and an honest account of what your experiment does
and does not establish. Well-analyzed negative results earn full credit.
