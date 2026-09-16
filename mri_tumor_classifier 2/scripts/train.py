#!/usr/bin/env python
"""End-to-end entry point: load config -> build data/model -> train -> evaluate.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Allow running this script directly (`python scripts/train.py`) without
# having installed the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_tumor import MRIDataset, build_model, evaluate_model, threshold, train_loop


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")

    root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]

    train_ds = MRIDataset(root=root, img_size=img_size, mode="train", seed=cfg["seed"])
    val_ds = MRIDataset(root=root, img_size=img_size, mode="val", seed=cfg["seed"])
    test_ds = MRIDataset(root=root, img_size=img_size, mode="test", seed=cfg["seed"])

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model = build_model(
        name=cfg["model"]["name"],
        in_channels=cfg["model"]["in_channels"],
        pretrained=cfg["model"].get("pretrained", True),
    )

    train_loop(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        wd=cfg["train"]["weight_decay"],
        patience=cfg["train"]["patience"],
        ckpt_path=cfg["train"]["ckpt_path"],
    )

    print("\nEvaluating on the held-out test set:")
    evaluate_model(model, test_loader, device=device, threshold_fn=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
