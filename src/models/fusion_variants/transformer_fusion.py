"""
transformer_fusion.py
Transformer-based multimodal fusion (Phase 5)
Logs accuracy, precision, recall, F1 for research tracking.
"""

import os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
MANIFEST = "data/custom/manifest_train.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS, BATCH_SIZE, LR = 20, 4, 1e-4

# ---------- Dataset ----------
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


# ---------- Model ----------
class TransformerFusion(nn.Module):
    def __init__(self, tdim=768, adim=768, vdim=1000, hidden=512, classes=3, n_heads=4, n_layers=2):
        super().__init__()
        self.text_fc = nn.Linear(tdim, hidden)
        self.audio_fc = nn.Linear(adim, hidden)
        self.video_fc = nn.Linear(vdim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, t, a, v):
        t = self.text_fc(t).unsqueeze(0)
        a = self.audio_fc(a).unsqueeze(0)
        v = self.video_fc(v).unsqueeze(0)
        seq = torch.cat([t, a, v], dim=0)
        fused = self.transformer(seq).mean(0)
        fused = self.drop(fused)
        return self.classifier(fused)


# ---------- Training / Evaluation ----------
def evaluate(model, loader, crit):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for t,a,v,lbl in loader:
            t,a,v,lbl = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE),lbl.to(DEVICE)
            out = model(t,a,v)
            loss = crit(out,lbl)
            total_loss += loss.item()
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)*100
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return total_loss/len(loader), acc, p*100, r*100, f1*100


def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss, correct = 0, 0
    for t,a,v,lbl in loader:
        t,a,v,lbl = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE),lbl.to(DEVICE)
        opt.zero_grad()
        out = model(t,a,v)
        loss = crit(out,lbl)
        loss.backward(); opt.step()
        total_loss += loss.item()
        correct += (out.argmax(1)==lbl).sum().item()
    acc = 100*correct/len(loader.dataset)
    return total_loss/len(loader), acc


# ---------- Main ----------
if __name__=="__main__":
    data = MultimodalDataset(MANIFEST, DATA_DIR)
    n = len(data)
    train_n = int(0.8*n)
    val_n = n - train_n
    train_ds, val_ds = random_split(data, [train_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TransformerFusion().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 Training Transformer Fusion on {DEVICE}")
    log = []

    for ep in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, crit)
        vl_loss, vl_acc, vl_p, vl_r, vl_f1 = evaluate(model, val_dl, crit)
        log.append([ep+1,tr_acc,vl_acc,vl_p,vl_r,vl_f1])
        print(f"Epoch {ep+1:02d}/{EPOCHS} | Train {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | F1 {vl_f1:.2f}%")

    torch.save(model.state_dict(), os.path.join(OUT_DIR,"transformer_fusion.pt"))
    df = pd.DataFrame(log, columns=["epoch","train_acc","val_acc","precision","recall","f1"])
    df.to_csv(os.path.join(OUT_DIR,"transformer_fusion_log.csv"),index=False)

    plt.plot(df["epoch"],df["val_acc"],label="Val Accuracy")
    plt.plot(df["epoch"],df["f1"],label="Val F1")
    plt.xlabel("Epoch"); plt.ylabel("Metric (%)")
    plt.title("Transformer Fusion Performance")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR,"transformer_fusion_curve.png"))
    plt.show()
