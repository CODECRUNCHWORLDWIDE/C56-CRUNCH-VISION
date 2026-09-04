# Exercise 1 — Build a group-aware, leakage-free Dataset

**Goal:** a leakage-free data pipeline from real image folders, with the group structure handled
correctly and the splitting treated as experiment design.

## Tasks

1. Take a real image-folder dataset (an Oxford-IIIT Pets subset, or your own labeled photos). Build a `Dataset`
   (or use `ImageFolder`) that loads images and integer labels; print the per-class counts and note any
   imbalance.
2. **Identify the group unit.** Determine what a "group" is in your data (an individual animal, a photo session,
   a source URL, a patient). If images cluster, extract a group id per image. Split into train/val/test **by
   group** using `GroupShuffleSplit`/`GroupKFold` so no group spans two splits. If your data has no group
   structure, argue in a comment why a per-image split is safe.
3. Apply *augmented* transforms to training and *clean* transforms to val/test. Compute normalization statistics
   on the **training split only** (or use ImageNet stats for a pretrained backbone), and apply the fixed
   transform to all splits.
4. **Leakage audit.** Programmatically confirm (a) no image path appears in more than one split, and (b) no
   group id appears in more than one split. Display a batch of augmented training images with labels to
   eyeball the transforms.

## Deliverable

A notebook with working DataLoaders, the group-aware split, the train-only normalization, a displayed augmented
batch, and an explicit assertion that neither images nor groups overlap across splits. Two sentences naming the
group unit and how your split prevents the invisible leak.
