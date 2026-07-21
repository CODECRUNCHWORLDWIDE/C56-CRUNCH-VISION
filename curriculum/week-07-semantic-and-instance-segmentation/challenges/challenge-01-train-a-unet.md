# Challenge 1 — Train a U-Net on a small dataset

Build the segmentation workhorse and see skip connections earn their keep.

1. Take a small semantic-segmentation dataset with masks (e.g. Oxford Pets segmentation, a medical set like a lung/cell dataset, or synthetic shapes you generate with known masks).
2. Implement or adapt a U-Net (encoder, decoder, skip connections). Train it with a Dice or cross-entropy (or combined) loss.
3. Report mIoU/Dice on a held-out split and overlay predictions.
4. **Ablation:** remove the skip connections and retrain. Show quantitatively (mIoU) and visually (blurrier boundaries) how much skips matter.

**Deliverable:** a trained U-Net with held-out mIoU/Dice, mask overlays, and the skip-connection ablation. Seeing masks degrade without skips makes the architecture's core idea concrete and memorable.
