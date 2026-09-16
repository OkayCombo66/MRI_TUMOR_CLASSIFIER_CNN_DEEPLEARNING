# Building an MRI Brain Tumor Classifier: From Tutorial Code to a Working Pipeline

## Why this project

I wanted hands-on experience with a real (if small) medical imaging problem: given an MRI scan, predict whether it shows a tumor. It's a classic binary image classification setup, but working with medical images — small datasets, class imbalance, the stakes of getting it wrong — makes it a more interesting exercise than a generic image classifier.

I started from a CNN/PyTorch tutorial to get the basic pieces in place: a `Dataset` class to load images, a small convolutional network, a training loop, and an evaluation script.

## What the project does

Given an MRI scan, the model outputs a probability that the scan shows a tumor. The pipeline:

1. **Data loading** (`dataset.py`) — reads MRI images from `yes/` (tumor) and `no/` (healthy) folders, resizes and normalizes them, and produces a stratified 70/20/10 train/val/test split.
2. **Model** (`model.py`) — a small 4-block CNN trained from scratch, with a ResNet-18 transfer-learning variant available as a second option, selectable from a config file rather than by editing code.
3. **Training** (`train.py`) — a standard training loop with early stopping on validation loss and automatic checkpointing of the best model.
4. **Evaluation** (`evaluate.py`) — accuracy, F1, and ROC-AUC on a held-out test set, plus a confusion matrix.

## Getting the tutorial code to actually run

The version I started from didn't run end-to-end — which is a pretty normal state for code copied out of a tutorial video or notebook, where snippets get pasted in a different order than they were written, or a variable gets renamed in one place and not another. Getting it to a working state meant going through each file and fixing what was actually broken, rather than assuming it worked because it *looked* plausible:

- **Dataset loading** — the image-resize helper referenced a `cv2` call that doesn't exist, the "healthy" image loop was accidentally appending into the "tumor" list (a copy-paste bug that would have silently mislabeled half the training data), and a block of accumulation logic had drifted inside the wrong function due to indentation, making it unreachable dead code.
- **Training loop** — the optimizer was called with a misspelled keyword argument, the model was never actually moved to the training device, and the early-stopping/best-checkpoint logic referenced variables that were never initialized.
- **Model definitions** — the ResNet-18 baseline referenced an import alias that didn't exist, and the file downloaded pretrained ImageNet weights automatically on import, which happens whether or not you were going to use that model.
- **Config vs. code mismatch** — the example config file described a directory layout (`train_dir` / `val_dir`) that the dataset loader had no code path for at all; it actually expects one root folder split internally. I made the config match what the code really does, instead of leaving a config option that silently does nothing.

None of these are exotic bugs — they're exactly the kind of thing that slips through when you're following along with a tutorial and adapting it on the fly. The useful lesson wasn't the individual fixes, it was the habit behind them: run the code, don't just read it, before trusting that it works.

## What I changed structurally

Beyond fixing bugs, I reorganized the project from a flat pile of scripts into a package (`mri_tumor/`) plus a config-driven CLI (`scripts/train.py --config configs/default.yaml`), so that swapping the model, dataset path, or training hyperparameters doesn't require editing source files. I also added a `.gitignore` for data and checkpoint artifacts, and split the README into setup/usage instructions with this write-up covering the reasoning and trade-offs separately.

## Evaluation approach — and what I'm deliberately not claiming

`evaluate.py` reports accuracy, F1, and ROC-AUC on a held-out test split, plus a confusion matrix. I'm intentionally not publishing specific numbers here: the public dataset this is built around is small (a few hundred images total), so any single run's accuracy is noisy and not something I'd stand behind as a stable, reproducible result without multiple runs and a proper hold-out strategy. Anyone running `scripts/train.py` against the dataset will get real numbers for their own run — that felt more honest than quoting a figure that would just be one lucky (or unlucky) seed.

## Limitations

This is a learning project, not a diagnostic tool, and I want to be upfront about why:

- The dataset is small and comes from a single public source — there's no evidence this generalizes to scans from a different scanner or hospital.
- There's no radiologist-reviewed ground truth beyond the dataset's own labels.
- It's binary tumor/no-tumor only, with no localization or tumor-type distinction.
- No systematic hyperparameter search was done — the defaults are a reasonable starting point, not a tuned result.

## What I'd do next

Given more time, the next steps I'd prioritize are: k-fold cross-validation instead of a single split (to get a real sense of variance given how small the dataset is), basic data augmentation to reduce overfitting risk, and a Grad-CAM style visualization so predictions come with some indication of *where* the model is looking in the image — useful both for debugging and for making the model's reasoning inspectable.
