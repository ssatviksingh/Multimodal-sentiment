"""
run_ablation_study.py

Journal-style ablation study runner.

This script:
- Loads a set of YAML configs describing ablation variants
- Reuses the HybridFusion feature dataset + epoch_pass utilities
- Trains/evaluates:
    - Text-only
    - Audio-only
    - Video-only
    - Early fusion
    - Late fusion
    - Hybrid transformer fusion (existing model)
- Collects accuracy / precision / recall / macro-F1
- Saves a summary CSV under research_extensions/results/

NOTE: This file lives entirely under research_extensions/ and does not
modify or replace the original training pipelines.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd

from src.models.fusion_variants.hybrid_fusion_vit_ast import (
    MultimodalDataset,
    HybridFusionModel,
    epoch_pass,
    DEVICE,
    SEED,
)


# ----------------- CONFIG STRUCTURE -----------------

@dataclass
class AblationConfig:
    experiment_name: str
    model_type: str
    manifest_path: str
    feature_dir: str
    out_dir: str
    epochs: int = 20
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-5
    seed: int = SEED
    num_classes: int = 3


def load_config(path: str) -> AblationConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Defensive casting in case YAML parses numerics as strings
    if "epochs" in raw:
        raw["epochs"] = int(raw["epochs"])
    if "batch_size" in raw:
        raw["batch_size"] = int(raw["batch_size"])
    if "lr" in raw:
        raw["lr"] = float(raw["lr"])
    if "weight_decay" in raw:
        raw["weight_decay"] = float(raw["weight_decay"])
    if "seed" in raw:
        raw["seed"] = int(raw["seed"])
    if "num_classes" in raw:
        raw["num_classes"] = int(raw["num_classes"])

    return AblationConfig(**raw)


# ----------------- MODEL VARIANTS -----------------

class TextOnlyClassifier(nn.Module):
    """Simple MLP classifier over text embeddings."""

    def __init__(self, text_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, t, a, v):
        return self.net(t)


class AudioOnlyClassifier(nn.Module):
    """Simple MLP classifier over audio embeddings."""

    def __init__(self, audio_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, t, a, v):
        return self.net(a)


class VideoOnlyClassifier(nn.Module):
    """Simple MLP classifier over video embeddings."""

    def __init__(self, video_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, t, a, v):
        return self.net(v)


class EarlyFusionClassifier(nn.Module):
    """Early fusion via concatenation of modality embeddings."""

    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, num_classes: int):
        super().__init__()
        fused_dim = text_dim + audio_dim + video_dim
        self.net = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, t, a, v):
        fused = torch.cat([t, a, v], dim=-1)
        return self.net(fused)


class LateFusionClassifier(nn.Module):
    """
    Late fusion:
    - Independent linear heads per modality
    - Learnable scalar weights combining logits
    """

    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, num_classes: int):
        super().__init__()
        self.head_t = nn.Linear(text_dim, num_classes)
        self.head_a = nn.Linear(audio_dim, num_classes)
        self.head_v = nn.Linear(video_dim, num_classes)

        # Learnable fusion weights
        self.logits_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32))

    def forward(self, t, a, v):
        log_t = self.head_t(t)
        log_a = self.head_a(a)
        log_v = self.head_v(v)
        w = torch.softmax(self.logits_weights, dim=0)
        return w[0] * log_t + w[1] * log_a + w[2] * log_v


def build_model(cfg: AblationConfig, text_dim: int, audio_dim: int, video_dim: int) -> nn.Module:
    """Factory for different ablation models."""
    mt = cfg.model_type.lower()
    if mt == "text_only":
        return TextOnlyClassifier(text_dim=text_dim, num_classes=cfg.num_classes)
    if mt == "audio_only":
        return AudioOnlyClassifier(audio_dim=audio_dim, num_classes=cfg.num_classes)
    if mt == "video_only":
        return VideoOnlyClassifier(video_dim=video_dim, num_classes=cfg.num_classes)
    if mt == "early_fusion":
        return EarlyFusionClassifier(
            text_dim=text_dim, audio_dim=audio_dim, video_dim=video_dim, num_classes=cfg.num_classes
        )
    if mt == "late_fusion":
        return LateFusionClassifier(
            text_dim=text_dim, audio_dim=audio_dim, video_dim=video_dim, num_classes=cfg.num_classes
        )
    if mt == "hybrid_fusion":
        # Reuse the existing HybridFusion model
        return HybridFusionModel(
            text_dim=text_dim, audio_dim=audio_dim, video_dim=video_dim, num_classes=cfg.num_classes
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


# ----------------- TRAIN / EVAL LOOP -----------------

def run_single_ablation(cfg: AblationConfig) -> Dict[str, Any]:
    """
    Train/eval one ablation configuration.
    Returns a dict of best validation metrics.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Dataset + grouped split (reusing the HybridFusion dataset)
    # To avoid leakage, we split by underlying clip/session, grouping rows
    # by a stable column such as audio_path (fallbacks to video_path/filename).
    full_ds = MultimodalDataset(cfg.manifest_path, cfg.feature_dir)

    manifest_df = pd.read_csv(cfg.manifest_path)
    manifest_df.columns = [c.strip().lower() for c in manifest_df.columns]

    if "audio_path" in manifest_df.columns:
        group_col = "audio_path"
    elif "video_path" in manifest_df.columns:
        group_col = "video_path"
    else:
        group_col = "filename"

    groups = manifest_df[group_col].astype(str)
    unique_groups = groups.drop_duplicates().sample(frac=1.0, random_state=cfg.seed)

    n_groups = len(unique_groups)
    n_train_groups = int(0.8 * n_groups)
    train_group_ids = set(unique_groups.iloc[:n_train_groups])
    val_group_ids = set(unique_groups.iloc[n_train_groups:])

    train_indices = [i for i, g in enumerate(groups) if g in train_group_ids]
    val_indices = [i for i, g in enumerate(groups) if g in val_group_ids]

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    train_n = len(train_indices)
    val_n = len(val_indices)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Peek one batch to infer embedding dims
    sample_batch = next(iter(train_dl))
    t_sample, a_sample, v_sample, _ = sample_batch
    text_dim = t_sample.shape[-1]
    audio_dim = a_sample.shape[-1]
    video_dim = v_sample.shape[-1]

    model = build_model(cfg, text_dim, audio_dim, video_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    best_metrics = {
        "val_acc": 0.0,
        "val_prec": 0.0,
        "val_rec": 0.0,
        "val_f1": 0.0,
    }

    logs: List[List[float]] = []

    print(f"\n===== Running ablation: {cfg.experiment_name} ({cfg.model_type}) =====")
    print(f"Train samples: {train_n}, Val samples: {val_n}")

    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = epoch_pass(model, train_dl, optimizer, criterion, train=True)
        vl_loss, vl_acc, vl_p, vl_r, vl_f1 = epoch_pass(model, val_dl, optimizer, criterion, train=False)

        logs.append(
            [ep, tr_loss, tr_acc, tr_p, tr_r, tr_f1, vl_loss, vl_acc, vl_p, vl_r, vl_f1]
        )

        print(
            f"[{cfg.experiment_name}] Epoch {ep}/{cfg.epochs} | "
            f"Train Acc {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | Val F1 {vl_f1:.2f}%"
        )

        # Track best epoch by validation macro-F1
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_metrics = {
                "val_acc": vl_acc,
                "val_prec": vl_p,
                "val_rec": vl_r,
                "val_f1": vl_f1,
            }
            # Save best checkpoint per setting
            ckpt_path = os.path.join(cfg.out_dir, f"{cfg.experiment_name}_best.pt")
            torch.save(model.state_dict(), ckpt_path)

    # Save per-epoch log for this setting
    cols = [
        "epoch",
        "tr_loss",
        "tr_acc",
        "tr_prec",
        "tr_rec",
        "tr_f1",
        "val_loss",
        "val_acc",
        "val_prec",
        "val_rec",
        "val_f1",
    ]
    df = pd.DataFrame(logs, columns=cols)
    log_csv = os.path.join(cfg.out_dir, f"{cfg.experiment_name}_log.csv")
    df.to_csv(log_csv, index=False)

    return best_metrics


# ----------------- ENTRYPOINT -----------------

def main():
    base_cfg_dir = os.path.join("research_extensions", "configs")
    cfg_files = [
        "ablation_text_only.yaml",
        "ablation_audio_only.yaml",
        "ablation_video_only.yaml",
        "ablation_early_fusion.yaml",
        "ablation_late_fusion.yaml",
        "ablation_hybrid_fusion.yaml",
    ]

    summary_rows: List[Dict[str, Any]] = []

    for cfg_name in cfg_files:
        cfg_path = os.path.join(base_cfg_dir, cfg_name)
        cfg = load_config(cfg_path)
        metrics = run_single_ablation(cfg)

        summary_rows.append(
            {
                "experiment_name": cfg.experiment_name,
                "model_type": cfg.model_type,
                "val_accuracy": metrics["val_acc"],
                "val_precision": metrics["val_prec"],
                "val_recall": metrics["val_rec"],
                "val_macro_f1": metrics["val_f1"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("research_extensions/results", exist_ok=True)
    summary_path = "research_extensions/results/ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Ablation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

