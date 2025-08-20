import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# to work with dataset.py, mode should be "train" when calling this dataloader

def train_loop(model, train_loader, val_loader, device, epochs=500, lr=0.0001, wd=0.0001, count=20, best_path = "best_model.pt", plot=True):


    hist_train, hist_val = [], []

    model = model.to_device(device)
    optimizer = torch.optim.AdamW(model_parameters(), lr=lr, weight_decayed=wd)
    criterion = nn.BCEWithLogitsLoss()

    
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch['image'].to(device)
            y = batch['label'].to(device).float().view(-1)

            optimizer.zero_grad()
            logits = model(x).view(-1)         
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        epoch_train = float(np.mean(train_losses))
        hist_train.append(epoch_train)

        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:           
                x = batch['image'].to(device)
                y = batch['label'].to(device).float().view(-1)

                logits = model(x).view(-1)     
                loss   = criterion(logits, y)
                val_losses.append(loss.item())

        epoch_val = float(np.mean(val_losses))
        hist_val.append(epoch_val)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {epoch_train:.6f} | val {epoch_val:.6f}")

        if epoch_val < best_val - 0.0001:
            best_val = epoch_val
            best_epoch = epoch
            no_improve = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)

        else:
            no_improve += 1
            if no_improve >= count:
                print(f"Early stopping at epoch {epoch}. Best val loss {best_val:.6f} at epoch {best_epoch}.")
                break

    print(f"Best epoch: {best_epoch}  (val loss {best_val:.6f})  → saved to {ckpt_path}")

    if plot:
        plt.figure(figsize=(9, 5))
        plt.plot(hist_train, label="train loss")
        plt.plot(hist_val, label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training versus Validation Loss")
        plt.grid(True); plt.legend()
        plt.show()

    return hist_train, hist_val, model
    