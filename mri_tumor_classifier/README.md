# MRI Brain Tumor Classification (PyTorch)

A small binary image classifier that labels brain MRI scans as **tumor** or **no tumor**, built around a compact CNN (with a ResNet-18 transfer-learning baseline available as an alternative). Trained and evaluated on the public [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) dataset (Navoneel Chakrabarty, Kaggle).

> **Not a medical device.** This is a learning project. It has not been validated on a clinically representative dataset, has not been reviewed by a radiologist, and must not be used to inform any real diagnostic or treatment decision.

## Project structure

```
mri_tumor_classifier/
├── mri_tumor/            # importable package
│   ├── dataset.py        # MRIDataset: loads images, does the train/val/test split
│   ├── model.py           # CNN + ResNet-18 baseline, build_model() factory
│   ├── train.py           # training loop with early stopping
│   ├── evaluate.py        # accuracy / F1 / ROC-AUC + confusion matrix
│   └── utils.py            # threshold() helper
├── scripts/
│   ├── train.py           # CLI: train a model end-to-end from a config file
│   └── evaluate.py         # CLI: evaluate a saved checkpoint
├── configs/
│   └── default.yaml
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset and place it so the `yes/` (tumor) and `no/` (healthy) folders sit under `data/brain_tumor_dataset/`:

```
data/brain_tumor_dataset/
├── yes/*.jpg
└── no/*.jpg
```

(`data/` is git-ignored — this is a placement instruction, not something committed to the repo.)

## Usage

Train with the default config:

```bash
python scripts/train.py --config configs/default.yaml
```

This trains the model, saves the best checkpoint (by validation loss, with early stopping) to `checkpoints/`, writes a loss-curve plot and confusion matrix to `reports/`, and prints test-set accuracy / F1 / ROC-AUC at the end.

Evaluate a saved checkpoint on its own:

```bash
python scripts/evaluate.py --config configs/default.yaml --ckpt checkpoints/best_model.pt
```

Switch to the ResNet-18 baseline instead of the custom CNN by setting `model.name: "resnet18"` in the config.

## Model

The default model is a small 4-block CNN (Conv → BatchNorm → ReLU → MaxPool, doubling channels up to 128) followed by a two-layer classification head with dropout, trained on 128×128 RGB crops with `BCEWithLogitsLoss`. `model.py` also includes a ResNet-18 variant (ImageNet-pretrained by default) as a transfer-learning baseline for comparison, selectable from the config without touching any code.

## Evaluation

`evaluate.py` reports accuracy, F1, and ROC-AUC on the held-out test split, plus a saved confusion-matrix figure. This repo intentionally does not hard-code performance numbers here — run it on the dataset yourself (`python scripts/train.py`) to get real figures for your environment and data split; pasting someone else's numbers as if they were reproducible from a shrinking, non-standardized dataset would be misleading.

## Known limitations

- Small dataset (a few hundred images total in the public source dataset), so reported metrics will have high variance run-to-run.
- No independent, clinically-sourced test set — the train/val/test split all comes from the same source distribution, so it says nothing about generalization to scans from a different hospital, scanner, or patient population.
- Binary tumor/no-tumor only — no localization, no tumor-type classification.
- No hyperparameter search was performed; the defaults in `configs/default.yaml` are a reasonable starting point, not a tuned result.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Built while following an online CNN/PyTorch tutorial as a learning exercise, then reworked into a proper package structure with a config-driven CLI, bug fixes, and documentation.
