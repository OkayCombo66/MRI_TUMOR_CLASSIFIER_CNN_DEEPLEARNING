#!/usr/bin/env python
"""Evaluate a saved checkpoint on the held-out test split.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --ckpt checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_tumor import MRIDataset, build_model, evaluate_model, threshold


def main(config_path, ckpt_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")

    test_ds = MRIDataset(
        root=cfg["data"]["root"], img_size=cfg["data"]["img_size"], mode="test", seed=cfg["seed"]
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = build_model(
        name=cfg["model"]["name"],
        in_channels=cfg["model"]["in_channels"],
        pretrained=False,  # weights come from the checkpoint, not ImageNet
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    evaluate_model(model, test_loader, device=device, threshold_fn=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best_model.pt")
    args = parser.parse_args()
    main(args.config, args.ckpt)
