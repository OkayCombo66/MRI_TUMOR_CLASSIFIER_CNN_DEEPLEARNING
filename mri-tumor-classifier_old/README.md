
# MRI Brain Tumor (Binary) — PyTorch

This repo trains a  CNN to classify MRI brain images as **tumor** vs **no tumor**.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/train.py
```

## Structure
- `mri_brain_tumor/` — importable package (dataset, model, training utilities)
- `scripts/` — runnable scripts (train/eval/predict)
- `notebooks/` — experiments/EDA
- `data/` — local datasets (ignored by git)
- `runs/` — checkpoints, logs (ignored by git)

## Notes
- Adjust your **data paths** inside `mri_brain_tumor/data/mri.py` if needed.
- CNN architecture mirrors the original notebook (Tanh + AvgPool).
