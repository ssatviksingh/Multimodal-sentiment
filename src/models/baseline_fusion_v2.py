"""
baseline_fusion_v2.py
Adds validation + accuracy logging for research tracking (Phase 3.1)
"""

import os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
MANIFEST = "data/custom/manifest_train.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS, BATCH_SIZE, LR = 15, 4, 1e-4

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, feature_dir):
        self.data = pd.read_csv(manifest_path)
        self.feature_dir = feature_dir
        self.data.columns = [c.strip().lower() for c in self.data.columns]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sid = row.get("filename") or row.get("id") or row.get("sample_id")
        text = torch.load(os.path.join(self.feature_dir, "text", f"{sid}.pt")).float()
        audio = torch.load(os.path.join(self.feature_dir, "audio", f"{sid}.pt")).float()
        video = torch.load(os.path.join(self.feature_dir, "video", f"{sid}.pt")).float()
        label = torch.tensor(int(row["label"]))
        return text, audio, video, label

class BaselineFusion(nn.Module):
    def __init__(self, tdim=768, adim=768, vdim=1000, hdim=512, classes=3):
        super().__init__()
        self.fc1 = nn.Linear(tdim + adim + vdim, hdim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hdim, classes)
    def forward(self, t, a, v):
        x = torch.cat((t, a, v), dim=1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

def run_epoch(model, loader, opt, crit, train=True):
    model.train(train)
    total_loss, correct = 0, 0
    for t,a,v,lbl in loader:
        t,a,v,lbl = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE),lbl.to(DEVICE)
        if train: opt.zero_grad()
        out = model(t,a,v)
        loss = crit(out,lbl)
        if train:
            loss.backward(); opt.step()
        total_loss += loss.item()
        correct += (out.argmax(1)==lbl).sum().item()
    return total_loss/len(loader), correct/len(loader.dataset)

if __name__=="__main__":
    data = MultimodalDataset(MANIFEST, DATA_DIR)
    n = len(data)
    train_n = int(0.8 * n)
    val_n = n - train_n  # ensures sum = n exactly
    train_ds, val_ds = random_split(data, [train_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = BaselineFusion().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    logs = []
    print(f"🚀 Training with validation on {DEVICE}")

    for ep in range(EPOCHS):
        tr_loss,tr_acc = run_epoch(model,train_dl,opt,crit,True)
        vl_loss,vl_acc = run_epoch(model,val_dl,opt,crit,False)
        logs.append([ep+1,tr_acc*100,vl_acc*100])
        print(f"Epoch {ep+1}/{EPOCHS} | Train:{tr_acc*100:.2f}% | Val:{vl_acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(OUT_DIR,"baseline_fusion_v2.pt"))
    df = pd.DataFrame(logs,columns=["epoch","train_acc","val_acc"])
    df.to_csv(os.path.join(OUT_DIR,"baseline_accuracy_log.csv"),index=False)

    plt.plot(df["epoch"],df["train_acc"],label="Train Acc")
    plt.plot(df["epoch"],df["val_acc"],label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Baseline Fusion V2 Accuracy Progress")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR,"baseline_v2_curve.png"))
    plt.show()
