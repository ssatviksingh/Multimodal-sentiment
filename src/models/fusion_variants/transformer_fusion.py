"""
transformer_fusion.py
Transformer-based multimodal fusion — same train/val manifests as hybrid.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.fusion_variants.training_cli import (
    build_fusion_train_parser,
    resolve_device,
    resolve_train_config,
    apply_out_dir,
)
from src.models.fusion_variants.feature_io import dataloader_kwargs, load_feature_pt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
TRAIN_MANIFEST = "data/manifest_train.csv"
VAL_MANIFEST = "data/manifest_val.csv"
OUT_DIR = "results"

EPOCHS, BATCH_SIZE, LR = 10, 4, 1e-4
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, feature_dir, max_samples=None):
        self.data = pd.read_csv(manifest_path)
        self.feature_dir = feature_dir
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        if max_samples is not None and len(self.data) > max_samples:
            self.data = self.data.iloc[:max_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sid = str(row.get("filename") or row.get("id") or row.get("sample_id"))
        t = load_feature_pt(os.path.join(self.feature_dir, "text", f"{sid}.pt")).squeeze()
        a = load_feature_pt(os.path.join(self.feature_dir, "audio", f"{sid}.pt")).squeeze()
        v = load_feature_pt(os.path.join(self.feature_dir, "video", f"{sid}.pt")).squeeze()
        if t.dim() > 1:
            t = t.mean(dim=0)
        if a.dim() > 1:
            a = a.mean(dim=0)
        if v.dim() > 1:
            v = v.mean(dim=0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return t, a, v, label


class TransformerFusion(nn.Module):
    def __init__(self, tdim=768, adim=768, vdim=768, hidden=512, classes=3, n_heads=4, n_layers=2):
        super().__init__()
        self.text_fc = nn.Linear(tdim, hidden)
        self.audio_fc = nn.Linear(adim, hidden)
        self.video_fc = nn.Linear(vdim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2)
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


def evaluate(model, loader, crit):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for t, a, v, lbl in loader:
            t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
            out = model(t, a, v)
            loss = crit(out, lbl)
            total_loss += loss.item()
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds) * 100
    _, _, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, acc, f1_macro * 100


def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss, correct = 0, 0
    n = 0
    for t, a, v, lbl in loader:
        t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
        opt.zero_grad()
        out = model(t, a, v)
        loss = crit(out, lbl)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == lbl).sum().item()
        n += lbl.size(0)
    acc = 100 * correct / max(n, 1)
    return total_loss / max(len(loader), 1), acc


def main():
    global DEVICE
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = build_fusion_train_parser("Transformer multimodal fusion training")
    args = parser.parse_args()
    DEVICE = resolve_device(args.device)
    epochs, batch_size, max_tr, max_va = resolve_train_config(args, EPOCHS, BATCH_SIZE)
    apply_out_dir(args.out_dir)
    out_dir = args.out_dir

    print(
        f"📂 TransformerFusion train: {args.train_manifest} | val: {args.val_manifest} "
        f"| device={DEVICE} | epochs={epochs} batch={batch_size} "
        f"max_train={max_tr} max_val={max_va} smoke={args.smoke}"
    )
    train_ds = MultimodalDataset(args.train_manifest, args.data_dir, max_samples=max_tr)
    val_ds = MultimodalDataset(args.val_manifest, args.data_dir, max_samples=max_va)

    dl_kw = dataloader_kwargs(DEVICE)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    model = TransformerFusion().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 Training Transformer Fusion on {DEVICE}")
    log = []

    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, crit)
        vl_loss, vl_acc, vl_f1_macro = evaluate(model, val_dl, crit)
        log.append([ep + 1, tr_loss, tr_acc, vl_loss, vl_acc, vl_f1_macro])
        print(
            f"Epoch {ep+1:02d}/{epochs} | Train {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | Val macro-F1 {vl_f1_macro:.2f}%"
        )

    torch.save(model.state_dict(), os.path.join(out_dir, "transformer_fusion.pt"))
    df = pd.DataFrame(
        log,
        columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro"],
    )
    df.to_csv(os.path.join(out_dir, "transformer_fusion_log.csv"), index=False)

    plt.plot(df["epoch"], df["val_acc"], label="Val Accuracy")
    plt.plot(df["epoch"], df["val_f1_macro"], label="Val macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (%)")
    plt.title("Transformer Fusion Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "transformer_fusion_curve.png"))
    plt.close()
    print("✅ Saved transformer_fusion_log.csv")


if __name__ == "__main__":
    main()
