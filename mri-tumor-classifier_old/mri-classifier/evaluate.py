import torch, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# to work with dataset.py, mode should be "val" when calling this dataloader

def evaluate_model(model, dataloader, device, threshold_fn, cm_path="reports/confusion_matrix.png"):
    model.eval()
    logits_all, y_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).view(-1)

            logits = model(x).view(-1)
            logits_all.append(logits.cpu())
            y_all.append(y.cpu())

    # flatten everything into numpy arrays
    logits = torch.cat(logits_all).numpy()
    y_true = torch.cat(y_all).numpy()

    # convert logits → probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # apply your custom threshold
    preds = threshold_fn(probs)

    # metrics
    print(f"Accuracy: {accuracy_score(y_true, preds):.4f}")
    print(f"F1      : {f1_score(y_true, preds):.4f}")
    print(f"ROC-AUC : {roc_auc_score(y_true, probs):.4f}")

    # confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i,j]), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(cm_path); plt.close()

    return {"accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds),
            "auc": roc_auc_score(y_true, probs),
            "cm": cm}