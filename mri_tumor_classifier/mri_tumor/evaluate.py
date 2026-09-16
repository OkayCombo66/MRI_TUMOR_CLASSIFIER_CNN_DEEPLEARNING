"""Evaluation: accuracy / F1 / ROC-AUC + confusion matrix figure."""

import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt

# Expects a dataloader built from MRIDataset with mode="val" or mode="test".


def evaluate_model(model, dataloader, device, threshold_fn, cm_path="reports/confusion_matrix.png"):
    model.eval()
    logits_all, y_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device).float()
            y = batch["label"].to(device).view(-1)

            logits = model(x).view(-1)
            logits_all.append(logits.cpu())
            y_all.append(y.cpu())

    logits = torch.cat(logits_all).numpy()
    y_true = torch.cat(y_all).numpy()

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = threshold_fn(probs)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "auc": roc_auc_score(y_true, probs),
    }
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1      : {metrics['f1']:.4f}")
    print(f"ROC-AUC : {metrics['auc']:.4f}")

    cm = confusion_matrix(y_true, preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(cm_path) or ".", exist_ok=True)
    plt.savefig(cm_path)
    plt.close()

    metrics["cm"] = cm
    return metrics
