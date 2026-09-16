"""Training loop with early stopping on validation loss."""

import os

import numpy as np
import torch
import torch.nn as nn

# Expects dataloaders built from MRIDataset with mode="train" / mode="val".


def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    epochs=500,
    lr=1e-4,
    wd=1e-4,
    patience=20,
    ckpt_path="checkpoints/best_model.pt",
    plot=True,
    plot_path="reports/loss_curve.png",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()

    hist_train, hist_val = [], []
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["image"].to(device).float()
            y = batch["label"].to(device).float().view(-1)

            optimizer.zero_grad()
            logits = model(x).view(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        epoch_train = float(np.mean(train_losses))
        hist_train.append(epoch_train)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device).float()
                y = batch["label"].to(device).float().view(-1)

                logits = model(x).view(-1)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

        epoch_val = float(np.mean(val_losses))
        hist_val.append(epoch_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {epoch_train:.6f} | val {epoch_val:.6f}")

        if epoch_val < best_val - 1e-4:
            best_val = epoch_val
            best_epoch = epoch
            no_improve = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val loss {best_val:.6f} at epoch {best_epoch}.")
                break

    print(f"Best epoch: {best_epoch}  (val loss {best_val:.6f})  -> saved to {ckpt_path}")

    # Training ends on the last epoch's weights, which are not the best ones.
    # Restore the checkpoint so the returned model matches what was saved.
    if best_epoch > 0:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Restored best weights from epoch {best_epoch}.")

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        plt.plot(hist_train, label="train loss")
        plt.plot(hist_val, label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs. validation loss")
        plt.grid(True)
        plt.legend()
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

    return hist_train, hist_val, model
