"""
hybrid_fusion_vit_ast.py

Hybrid Transformer fusion:
- Projects existing modality embeddings (text/audio/video) to a shared dim
- Cross-modal Transformer encoder across modalities (FFN width 4× proj dim)
- Modality-specific classifiers + gated ensemble; auxiliary CE on shared + modality heads
- Class-weighted loss (inverse frequency), label smoothing, gradient clipping
- Logs macro precision/recall/F1 each epoch; saves CSV + PNG + packaged checkpoint

Train/val: data/manifest_train.csv, data/manifest_val.csv.
Test: data/manifest_test.csv via evaluate_model.py (load_hybrid_for_eval supports old/new checkpoints).

Features: data/features/{text,audio,video}/*.pt
"""

from __future__ import annotations

import math
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.fusion_variants.training_cli import (
    build_fusion_train_parser,
    resolve_device,
    resolve_train_config,
    apply_out_dir,
)
from src.models.fusion_variants.feature_io import dataloader_kwargs, load_checkpoint_dict, load_feature_pt

# -------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
TRAIN_MANIFEST = "data/manifest_train.csv"
VAL_MANIFEST = "data/manifest_val.csv"
OUT_DIR = "results"

EPOCHS = 12
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
SEED = 42
# Auxiliary CE on shared + per-modality heads (helps frozen-embedding fusion)
AUX_LOSS_WEIGHT = 0.2
GRAD_CLIP_NORM = 1.0

torch.manual_seed(SEED)
np.random.seed(SEED)


def infer_num_classes(manifest_path: str) -> int:
    df = pd.read_csv(manifest_path)
    df.columns = [c.strip().lower() for c in df.columns]
    labels = df["label"].astype(int)
    return int(labels.max()) + 1


def class_weights_tensor(train_labels: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    """Inverse-frequency weights (mean ~1); stabilizes imbalanced 3-class."""
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    w = len(train_labels) / (num_classes * np.maximum(counts, 1.0))
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32, device=device)


# -------- DATASET ----------
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

        try:
            t = load_feature_pt(os.path.join(self.feature_dir, "text", f"{sid}.pt")).squeeze()
            a = load_feature_pt(os.path.join(self.feature_dir, "audio", f"{sid}.pt")).squeeze()
            v = load_feature_pt(os.path.join(self.feature_dir, "video", f"{sid}.pt")).squeeze()

            if t.dim() > 1:
                t = t.mean(dim=0)
            if a.dim() > 1:
                a = a.mean(dim=0)
            if v.dim() > 1:
                v = v.mean(dim=0)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ Missing one or more feature files for sample '{sid}' → {e.filename}")

        lbl = torch.tensor(int(row["label"]), dtype=torch.long)
        return t, a, v, lbl


# -------- MODEL ----------
class HybridFusionModel(nn.Module):
    def __init__(
        self,
        text_dim=768,
        audio_dim=768,
        video_dim=768,
        proj_dim=512,
        transformer_layers=2,
        nhead=8,
        hidden=512,
        num_classes=3,
        dim_feedforward: int | None = None,
    ):
        super().__init__()
        self.proj_t = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2))
        self.proj_a = nn.Sequential(nn.Linear(audio_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2))
        self.proj_v = nn.Sequential(nn.Linear(video_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2))

        self.pos_emb = nn.Parameter(torch.randn(1, 3, proj_dim) * 0.02)

        if dim_feedforward is not None:
            ff = dim_feedforward
        else:
            ff = max(hidden, 4 * proj_dim)
        self._dim_feedforward = ff
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=0.2,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.classifier_shared = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

        self.class_t = nn.Linear(proj_dim, num_classes)
        self.class_a = nn.Linear(proj_dim, num_classes)
        self.class_v = nn.Linear(proj_dim, num_classes)

        self.gate_logits = nn.Parameter(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32))
        # Learned blend shared vs gated modality logits (sigmoid ≈ previous 0.6 / 0.4 at init)
        self.fusion_logit = nn.Parameter(torch.tensor(math.log(0.6 / 0.4)))

    def forward(self, t, a, v, return_components=False):
        pt = self.proj_t(t)
        pa = self.proj_a(a)
        pv = self.proj_v(v)

        if self.training:
            p_drop = float(getattr(self, "modality_dropout_p", 0.0))
            if p_drop > 0:
                B = pt.shape[0]
                device = pt.device
                drop = torch.rand(B, device=device) < p_drop
                which = torch.randint(0, 3, (B,), device=device)
                mask = torch.zeros(B, 3, device=device, dtype=pt.dtype)
                mask[torch.arange(B, device=device), which] = drop.to(pt.dtype)
                pt = pt * (1.0 - mask[:, 0:1])
                pa = pa * (1.0 - mask[:, 1:2])
                pv = pv * (1.0 - mask[:, 2:3])

        seq = torch.stack([pt, pa, pv], dim=1)
        seq = seq + self.pos_emb.to(seq.device)

        trans_out = self.transformer(seq)
        pooled = trans_out.mean(dim=1)

        shared_logits = self.classifier_shared(pooled)

        log_t = self.class_t(pt)
        log_a = self.class_a(pa)
        log_v = self.class_v(pv)

        gates = torch.softmax(self.gate_logits, dim=0)
        combined = gates[0] * log_t + gates[1] * log_a + gates[2] * log_v
        w = torch.sigmoid(self.fusion_logit)
        final = w * shared_logits + (1.0 - w) * combined

        if return_components:
            return final, shared_logits, (log_t, log_a, log_v), gates
        return final


def _infer_dim_feedforward_from_state_dict(sd: dict) -> int | None:
    w = sd.get("transformer.layers.0.linear1.weight")
    if w is not None:
        return int(w.shape[0])
    return None


def load_hybrid_for_eval(checkpoint_path: str, map_location):
    """Load HybridFusionModel from raw state_dict or packaged checkpoint dict."""
    raw = load_checkpoint_dict(checkpoint_path, map_location=map_location)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        dim_ff = raw.get("dim_feedforward")
        if dim_ff is None:
            dim_ff = _infer_dim_feedforward_from_state_dict(sd)
        kw = {
            "text_dim": raw.get("text_dim", 768),
            "audio_dim": raw.get("audio_dim", 768),
            "video_dim": raw.get("video_dim", 768),
            "num_classes": raw.get("num_classes", 3),
        }
        if dim_ff is not None:
            kw["dim_feedforward"] = dim_ff
        model = HybridFusionModel(**kw)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"ℹ️ Checkpoint load (non-strict): missing keys {missing} — using defaults for those.")
        if unexpected:
            print(f"ℹ️ Checkpoint load: unexpected keys {unexpected} (ignored).")
        return model
    sd = raw
    dim_ff = _infer_dim_feedforward_from_state_dict(sd)
    kw = {}
    if dim_ff is not None:
        kw["dim_feedforward"] = dim_ff
    model = HybridFusionModel(**kw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"ℹ️ Checkpoint load (non-strict): missing keys {missing} — using defaults for those.")
    if unexpected:
        print(f"ℹ️ Checkpoint load: unexpected keys {unexpected} (ignored).")
    return model


def _save_training_curve_png(out_dir: str, df: pd.DataFrame) -> None:
    """Save learning curve; tolerate Windows OSError 22 on some matplotlib/Pillow paths."""
    base = os.path.join(out_dir, "hybrid_fusion_curve.png")
    path = os.path.normpath(os.path.abspath(base))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["val_acc"], label="Val Acc")
    ax.plot(df["epoch"], df["val_f1_macro"], label="Val macro-F1")
    ax.plot(df["epoch"], df["tr_acc"], label="Train Acc", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric (%)")
    ax.set_title("Hybrid Fusion (ViT/AST-style) Performance")
    ax.legend()
    ax.grid(True)
    try:
        fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
    except OSError as e:
        alt = os.path.join(out_dir, "hybrid_curve.png")
        fig.savefig(os.path.normpath(os.path.abspath(alt)), format="png", dpi=150, bbox_inches="tight")
        print(f"⚠️ Saved curve to {alt} (primary path failed: {e})")
    finally:
        plt.close(fig)


# -------- UTILITIES ----------
def epoch_pass(model, loader, opt, crit, train=True, aux_weight: float = 0.0):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    all_preds = []
    all_labels = []

    for t, a, v, lbl in loader:
        t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
        if train:
            opt.zero_grad()
            if aux_weight > 0:
                final, shared, (log_t, log_a, log_v), _ = model(t, a, v, return_components=True)
                loss = crit(final, lbl)
                aux = (
                    crit(shared, lbl)
                    + crit(log_t, lbl)
                    + crit(log_a, lbl)
                    + crit(log_v, lbl)
                ) / 4.0
                loss = loss + aux_weight * aux
            else:
                out = model(t, a, v)
                loss = crit(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            out = final if aux_weight > 0 else out
        else:
            with torch.no_grad():
                out = model(t, a, v)
                loss = crit(out, lbl)
        losses.append(loss.item())
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbl.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(all_labels, all_preds) * 100.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, acc, p_macro * 100.0, r_macro * 100.0, f1_macro * 100.0


# -------- MAIN ----------
def main():
    global DEVICE
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = build_fusion_train_parser("HybridFusion (ViT/AST-style) training")
    args = parser.parse_args()
    DEVICE = resolve_device(args.device)
    epochs, batch_size, max_tr, max_va = resolve_train_config(args, EPOCHS, BATCH_SIZE)
    apply_out_dir(args.out_dir)
    out_dir = args.out_dir
    lr = args.lr if args.lr is not None else LR
    modality_dropout_p = float(args.modality_dropout)
    early_patience = int(args.early_stopping_patience or 0)

    print(
        f"📂 HybridFusion train: {args.train_manifest} | val: {args.val_manifest} "
        f"| device={DEVICE} | epochs={epochs} batch={batch_size} "
        f"max_train={max_tr} max_val={max_va} smoke={args.smoke} "
        f"| lr={lr:.2e} modality_dropout={modality_dropout_p} early_stop_patience={early_patience}"
    )
    train_ds = MultimodalDataset(args.train_manifest, args.data_dir, max_samples=max_tr)
    val_ds = MultimodalDataset(args.val_manifest, args.data_dir, max_samples=max_va)

    dl_kw = dataloader_kwargs(DEVICE)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    num_classes = infer_num_classes(args.train_manifest)
    train_labels = train_ds.data["label"].astype(int).values
    class_w = class_weights_tensor(train_labels, num_classes, DEVICE)
    model = HybridFusionModel(num_classes=num_classes).to(DEVICE)
    model.modality_dropout_p = modality_dropout_p
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    def step_scheduler(scheduler, metric, current_lr):
        old_lr = current_lr
        scheduler.step(metric)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"⚙️ LR reduced: {old_lr:.2e} → {new_lr:.2e}")

    logs = []
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    print(
        f"🚀 Training HybridFusion on {DEVICE} for {epochs} epochs "
        f"(classes={num_classes}, aux_loss={AUX_LOSS_WEIGHT})"
    )
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = epoch_pass(
            model, train_dl, opt, crit, train=True, aux_weight=AUX_LOSS_WEIGHT
        )
        vl_loss, vl_acc, vl_p, vl_r, vl_f1 = epoch_pass(
            model, val_dl, opt, crit, train=False, aux_weight=0.0
        )

        logs.append([ep, tr_loss, tr_acc, tr_p, tr_r, tr_f1, vl_loss, vl_acc, vl_p, vl_r, vl_f1])
        print(
            f"Epoch {ep}/{epochs} | Train Acc {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | "
            f"Val macro-F1 {vl_f1:.2f}%"
        )

        step_scheduler(scheduler, vl_f1, opt.param_groups[0]["lr"])

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch = ep
            epochs_no_improve = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "text_dim": 768,
                "audio_dim": 768,
                "video_dim": 768,
                "dim_feedforward": int(model._dim_feedforward),
            }
            torch.save(ckpt, os.path.join(out_dir, "hybrid_fusion_best.pt"))
        else:
            epochs_no_improve += 1
            if early_patience > 0 and epochs_no_improve >= early_patience:
                print(f"⏹ Early stopping: no val macro-F1 improvement for {early_patience} epochs (best epoch {best_epoch}).")
                break

    cols = [
        "epoch",
        "tr_loss",
        "tr_acc",
        "tr_prec",
        "tr_rec",
        "tr_f1_macro",
        "val_loss",
        "val_acc",
        "val_prec",
        "val_rec",
        "val_f1_macro",
    ]
    df = pd.DataFrame(logs, columns=cols)
    df.to_csv(os.path.join(out_dir, "hybrid_fusion_log.csv"), index=False)

    _save_training_curve_png(out_dir, df)

    print(f"✅ Best val macro-F1: {best_val_f1:.2f}% at epoch {best_epoch}")
    print(f"Saved: {os.path.join(out_dir, 'hybrid_fusion_best.pt')} and hybrid_fusion_log.csv")


HybridFusion = HybridFusionModel


if __name__ == "__main__":
    main()
