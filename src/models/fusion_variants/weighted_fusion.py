"""
weighted_fusion.py
Weighted multimodal fusion model for sentiment analysis (Phase 4)
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

EPOCHS, BATCH_SIZE, LR = 20, 4, 1e-4

# ---------------- Dataset ----------------
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

# ---------------- Model ----------------
class WeightedFusion(nn.Module):
    def __init__(self, tdim=768, adim=768, vdim=1000, hdim=512, classes=3):
        super().__init__()
        self.text_fc = nn.Linear(tdim, hdim)
        self.audio_fc = nn.Linear(adim, hdim)
        self.video_fc = nn.Linear(vdim, hdim)

        # Learnable weights
        self.alpha_t = nn.Parameter(torch.tensor(0.33))
        self.alpha_a = nn.Parameter(torch.tensor(0.33))
        self.alpha_v = nn.Parameter(torch.tensor(0.34))

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(hdim, classes)

    def forward(self, t, a, v):
        t = self.relu(self.text_fc(t))
        a = self.relu(self.audio_fc(a))
        v = self.relu(self.video_fc(v))
        # Normalize weights so they sum to 1
        alphas = torch.softmax(torch.stack([self.alpha_t, self.alpha_a, self.alpha_v]), dim=0)
        fused = alphas[0]*t + alphas[1]*a + alphas[2]*v
        fused = self.drop(fused)
        return self.classifier(fused)

# ---------------- Train/Eval Loop ----------------
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

# ---------------- Main ----------------
if __name__=="__main__":
    data = MultimodalDataset(MANIFEST, DATA_DIR)
    n = len(data)
    train_n = int(0.8*n)
    val_n = n - train_n
    train_ds, val_ds = random_split(data, [train_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = WeightedFusion().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    logs = []
    print(f"🚀 Training Weighted Fusion on {DEVICE}")

    for ep in range(EPOCHS):
        tr_loss,tr_acc = run_epoch(model,train_dl,opt,crit,True)
        vl_loss,vl_acc = run_epoch(model,val_dl,opt,crit,False)
        logs.append([ep+1,tr_acc*100,vl_acc*100])
        print(f"Epoch {ep+1}/{EPOCHS} | Train:{tr_acc*100:.2f}% | Val:{vl_acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(OUT_DIR,"weighted_fusion.pt"))
    df = pd.DataFrame(logs,columns=["epoch","train_acc","val_acc"])
    df.to_csv(os.path.join(OUT_DIR,"weighted_fusion_log.csv"),index=False)

    plt.plot(df["epoch"],df["train_acc"],label="Train Acc")
    plt.plot(df["epoch"],df["val_acc"],label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Weighted Fusion Accuracy Progress")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR,"weighted_fusion_curve.png"))
    plt.show()
