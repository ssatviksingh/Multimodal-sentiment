"""
weighted_fusion.py
Weighted multimodal fusion — same train/val manifests as hybrid (data/manifest_train.csv, data/manifest_val.csv).
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


class WeightedFusion(nn.Module):
    def __init__(self, tdim=768, adim=768, vdim=768, hdim=512, classes=3):
        super().__init__()
        self.text_fc = nn.Linear(tdim, hdim)
        self.audio_fc = nn.Linear(adim, hdim)
        self.video_fc = nn.Linear(vdim, hdim)

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
        alphas = torch.softmax(torch.stack([self.alpha_t, self.alpha_a, self.alpha_v]), dim=0)
        fused = alphas[0] * t + alphas[1] * a + alphas[2] * v
        fused = self.drop(fused)
        return self.classifier(fused)


def run_epoch(model, loader, opt, crit, train=True):
    if train:
        model.train()
    else:
        model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for t, a, v, lbl in loader:
        t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
        if train:
            opt.zero_grad()
        out = model(t, a, v)
        loss = crit(out, lbl)
        if train:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        n_batches += 1
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbl.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    _, _, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, acc, f1_macro * 100.0


def main():
    global DEVICE
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = build_fusion_train_parser("Weighted multimodal fusion training")
    args = parser.parse_args()
    DEVICE = resolve_device(args.device)
    epochs, batch_size, max_tr, max_va = resolve_train_config(args, EPOCHS, BATCH_SIZE)
    apply_out_dir(args.out_dir)
    out_dir = args.out_dir

    print(
        f"📂 WeightedFusion train: {args.train_manifest} | val: {args.val_manifest} "
        f"| device={DEVICE} | epochs={epochs} batch={batch_size} "
        f"max_train={max_tr} max_val={max_va} smoke={args.smoke}"
    )
    train_ds = MultimodalDataset(args.train_manifest, args.data_dir, max_samples=max_tr)
    val_ds = MultimodalDataset(args.val_manifest, args.data_dir, max_samples=max_va)

    dl_kw = dataloader_kwargs(DEVICE)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    model = WeightedFusion().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    logs = []
    print(f"🚀 Training Weighted Fusion on {DEVICE}")

    for ep in range(epochs):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, opt, crit, train=True)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, val_dl, opt, crit, train=False)
        logs.append([ep + 1, tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1])
        print(
            f"Epoch {ep+1}/{epochs} | Train {tr_acc:.2f}% | Val {vl_acc:.2f}% | Val macro-F1 {vl_f1:.2f}%"
        )

    torch.save(model.state_dict(), os.path.join(out_dir, "weighted_fusion.pt"))
    df = pd.DataFrame(
        logs,
        columns=["epoch", "train_loss", "train_acc", "train_f1_macro", "val_loss", "val_acc", "val_f1_macro"],
    )
    df.to_csv(os.path.join(out_dir, "weighted_fusion_log.csv"), index=False)

    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.plot(df["epoch"], df["val_f1_macro"], label="Val macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (%)")
    plt.title("Weighted Fusion")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "weighted_fusion_curve.png"))
    plt.close()
    print("✅ Saved weighted_fusion_log.csv")


if __name__ == "__main__":
    main()
