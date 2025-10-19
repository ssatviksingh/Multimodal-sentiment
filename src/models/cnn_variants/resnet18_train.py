"""
resnet18_train.py — CNN baseline for multimodal benchmark
"""
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import models
import pandas as pd, numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "results"; os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

# Dummy dataset (replace later with your fused features)
X = torch.randn(600, 3, 224, 224)
y = torch.randint(0, 3, (600,))
train_size = int(0.8 * len(X))
train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, len(X) - train_size])
train_dl, val_dl = DataLoader(train_ds, 8, True), DataLoader(val_ds, 8)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def epoch_pass(dl, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train: opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        if train: loss.backward(); opt.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total * 100

logs = []
best_acc = 0
for ep in range(5):
    tr_loss, tr_acc = epoch_pass(train_dl, True)
    vl_loss, vl_acc = epoch_pass(val_dl, False)
    logs.append([ep+1, tr_loss, tr_acc, vl_loss, vl_acc])
    print(f"Epoch {ep+1}/5 | Train {tr_acc:.2f}% | Val {vl_acc:.2f}%")
    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "resnet18_best.pt"))

pd.DataFrame(logs, columns=["epoch","train_loss","train_acc","val_loss","val_acc"]).to_csv(
    os.path.join(OUT_DIR, "resnet18_log.csv"), index=False)
print("✅ Saved: resnet18_best.pt + resnet18_log.csv")
