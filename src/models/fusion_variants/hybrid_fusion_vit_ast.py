"""
hybrid_fusion_vit_ast.py

Hybrid Transformer fusion:
- Projects existing modality embeddings (text/audio/video) to a shared dim
- Cross-modal Transformer encoder across modalities
- Modality-specific classifiers + gated ensemble (learnable gates)
- Logs accuracy, precision, recall, F1 each epoch and saves CSV + PNG

Works with your current data/features (text/*.pt, audio/*.pt, video/*.pt).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
MANIFEST = "data/custom/manifest_train_expanded.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -------- DATASET ----------
class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, feature_dir):
        self.data = pd.read_csv(manifest_path)
        self.feature_dir = feature_dir
        self.data.columns = [c.strip().lower() for c in self.data.columns]

        # detect available files
        self.available = {
            "text": set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(feature_dir, "text")) if f.endswith(".pt")),
            "audio": set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(feature_dir, "audio")) if f.endswith(".pt")),
            "video": set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(feature_dir, "video")) if f.endswith(".pt"))
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sid = str(row.get("filename") or row.get("id") or row.get("sample_id"))

        # fallback: auto-map if filename not found
        # if sid not in self.available["text"]:
        #     try:
        #         sid = list(self.available["text"])[idx % len(self.available["text"])]
        #         print(f"⚠️ Using fallback ID '{sid}' instead of missing '{row.get('filename')}'")
        #     except Exception:
        #         raise FileNotFoundError(f"❌ Missing feature file for sample {sid}")

        try:
            t = torch.load(os.path.join(self.feature_dir, "text", f"{sid}.pt")).float().squeeze()
            a = torch.load(os.path.join(self.feature_dir, "audio", f"{sid}.pt")).float().squeeze()
            v = torch.load(os.path.join(self.feature_dir, "video", f"{sid}.pt")).float().squeeze()

            # Ensure all are 1D tensors
            if t.dim() > 1: t = t.mean(dim=0)
            if a.dim() > 1: a = a.mean(dim=0)
            if v.dim() > 1: v = v.mean(dim=0)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ Missing one or more feature files for sample '{sid}' → {e.filename}")

        lbl = int(row["label"])
        return t, a, v, lbl



# -------- MODEL ----------
class HybridFusionModel(nn.Module):
    def __init__(self,
                 text_dim=768, audio_dim=768, video_dim=768,
                 proj_dim=512, transformer_layers=2, nhead=8, hidden=512, num_classes=3):
        super().__init__()
                # modality projection to same dimension (use provided input dims)
        self.proj_t = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2))
        self.proj_a = nn.Sequential(nn.Linear(audio_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2))
        self.proj_v = nn.Sequential(nn.Linear(video_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2))

        # positional encoding for 3-token sequence (learned). Make shape (1, 3, P) so broadcasting is explicit.
        self.pos_emb = nn.Parameter(torch.randn(1, 3, proj_dim) * 0.02)

        # Transformer Encoder across modalities - use batch_first=True and expect (B, 3, P)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)


        # Shared classifier from transformer output
        self.classifier_shared = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

        # Modality-specific auxiliary classifiers (late fusion)
        self.class_t = nn.Linear(proj_dim, num_classes)
        self.class_a = nn.Linear(proj_dim, num_classes)
        self.class_v = nn.Linear(proj_dim, num_classes)

        # Gating network (learnable scalar per modality)
        self.gate_logits = nn.Parameter(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32))

    def forward(self, t, a, v, return_components=False):
        # project
        pt = self.proj_t(t)  # (B, P)
        pa = self.proj_a(a)
        pv = self.proj_v(v)

        # build sequence with batch as first dim: (B, 3, P)
        seq = torch.stack([pt, pa, pv], dim=1)  # (B, 3, P)
        seq = seq + self.pos_emb.to(seq.device)  # pos_emb: (1,3,P) broadcast to (B,3,P)

        # transformer (batch_first=True)
        trans_out = self.transformer(seq)  # (B, 3, P)
        pooled = trans_out.mean(dim=1)     # (B, P) -- mean across tokens


        # shared logits
        shared_logits = self.classifier_shared(pooled)  # (B, C)

        # auxiliary logits
        log_t = self.class_t(pt)
        log_a = self.class_a(pa)
        log_v = self.class_v(pv)

        # gates normalized
        gates = torch.softmax(self.gate_logits, dim=0)  # (3,)
        # combine logits: weighted average between shared and modality-specific
        combined = gates[0]*log_t + gates[1]*log_a + gates[2]*log_v
        # final: add shared logits (gives both cross-modal + per-modality)
        final = 0.6*shared_logits + 0.4*combined

        if return_components:
            return final, shared_logits, (log_t, log_a, log_v), gates
        return final

# -------- UTILITIES ----------
def epoch_pass(model, loader, opt, crit, train=True):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    all_preds = []
    all_labels = []

    for t,a,v,lbl in loader:
        t,a,v,lbl = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE),lbl.to(DEVICE)
        if train:
            opt.zero_grad()
        out = model(t,a,v)
        loss = crit(out, lbl)
        if train:
            loss.backward()
            opt.step()
        losses.append(loss.item())
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbl.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(all_labels, all_preds) * 100.0
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, p*100.0, r*100.0, f1*100.0

# -------- MAIN ----------
def main():
    data = MultimodalDataset(MANIFEST, DATA_DIR)
    n = len(data)
    train_n = int(0.8 * n)
    val_n = n - train_n
    train_ds, val_ds = random_split(data, [train_n, val_n], generator=torch.Generator().manual_seed(SEED))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = HybridFusionModel().to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)  # slight smoothing for robustness
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)


    def step_scheduler(scheduler, metric, current_lr):
        old_lr = current_lr
        scheduler.step(metric)
        new_lr = opt.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"⚙️ LR reduced: {old_lr:.2e} → {new_lr:.2e}")


    logs = []
    best_val_f1 = -1.0
    best_epoch = -1

    print(f"🚀 Training HybridFusion on {DEVICE} for {EPOCHS} epochs")
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = epoch_pass(model, train_dl, opt, crit, train=True)
        vl_loss, vl_acc, vl_p, vl_r, vl_f1 = epoch_pass(model, val_dl, opt, crit, train=False)

        logs.append([ep, tr_loss, tr_acc, tr_p, tr_r, tr_f1, vl_loss, vl_acc, vl_p, vl_r, vl_f1])
        print(f"Epoch {ep}/{EPOCHS} | Train Acc {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | Val F1 {vl_f1:.2f}%")

        step_scheduler(scheduler, vl_f1, opt.param_groups[0]['lr'])

        # save best by val_f1
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "hybrid_fusion_best.pt"))

    # save logs
    cols = ["epoch","tr_loss","tr_acc","tr_prec","tr_rec","tr_f1","val_loss","val_acc","val_prec","val_rec","val_f1"]
    df = pd.DataFrame(logs, columns=cols)
    df.to_csv(os.path.join(OUT_DIR, "hybrid_fusion_log.csv"), index=False)

    # plot metrics
    plt.figure(figsize=(9,5))
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.plot(df["epoch"], df["val_f1"], label="Val F1")
    plt.plot(df["epoch"], df["tr_acc"], label="Train Acc", alpha=0.6)
    plt.xlabel("Epoch"); plt.ylabel("Metric (%)")
    plt.title("Hybrid Fusion (ViT/AST-style) Performance")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "hybrid_fusion_curve.png"))
    plt.show()

    print(f"✅ Best val F1: {best_val_f1:.2f}% at epoch {best_epoch}")
    print("Saved: hybrid_fusion_best.pt and hybrid_fusion_log.csv")
    
    
# ✅ For external imports
HybridFusion = HybridFusionModel


if __name__ == "__main__":
    main()
