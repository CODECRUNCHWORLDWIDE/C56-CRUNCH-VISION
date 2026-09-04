# Challenge 1 — Train a U-Net and prove skip connections earn their keep

Build the segmentation workhorse and turn "skips matter" from a claim into a measured, visible result.

1. Take a small semantic-segmentation dataset with masks — Oxford-IIIT Pet segmentation, a medical set (a cell or lung
   dataset), or synthetic shapes you generate with known ground-truth masks (fully reproducible, recommended for the
   ablation).
2. Implement or adapt a U-Net (encoder, decoder, **concatenating** skip connections). Train with a combined **Dice + CE**
   loss (Lecture 4's robust default). Log train/val loss and val Dice/mIoU per epoch.
3. Report **mIoU and Dice on a held-out split**, per class, and overlay predictions on several validation images.
4. **The skip-connection ablation.** Retrain an identical network with the skip connections removed (decoder gets only
   the upsampled deep features). Compare *quantitatively* (mIoU/Dice drop) and *visually* (blurrier, boundary-leaking
   masks). Also add a **Boundary IoU** measurement to show the degradation is concentrated exactly where theory predicts —
   at the edges.
5. Write a short analysis connecting the measured drop to Lecture 2's argument (deep features know 'what' but lost
   'where'; skips restore 'where').

**Deliverable:** a trained U-Net with held-out mIoU/Dice and overlays, the skip vs. no-skip ablation with both a
metric table (including Boundary IoU) and a visual gallery, and a paragraph explaining the mechanism. Seeing masks
degrade at the boundaries without skips makes the architecture's core idea concrete and unforgettable.
