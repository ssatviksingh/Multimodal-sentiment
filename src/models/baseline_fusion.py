"""
baseline_fusion.py
------------------
Baseline multimodal sentiment classifier using concatenated embeddings
from text, audio, and video. Expected accuracy ≈ 60–65% initially.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------- #
#         CONFIGURATION         #
# ----------------------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
MANIFEST = "data/custom/manifest_train.csv"
EPOCHS = 15
BATCH_SIZE = 4
LR = 1e-4
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- #
#       DATASET LOADER          #
# ----------------------------- #

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, feature_dir):
        self.data = pd.read_csv(manifest_path)
        self.feature_dir = feature_dir

        # normalize column names (for safety)
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row.get("filename") or row.get("id") or row.get("sample_id")

        text_path = os.path.join(self.feature_dir, "text", f"{sample_id}.pt")
        audio_path = os.path.join(self.feature_dir, "audio", f"{sample_id}.pt")
        video_path = os.path.join(self.feature_dir, "video", f"{sample_id}.pt")

        text = torch.load(text_path).float()
        audio = torch.load(audio_path).float()
        video = torch.load(video_path).float()
        label = torch.tensor(int(row["label"]))
        return text, audio, video, label



# ----------------------------- #
#        MODEL DEFINITION       #
# ----------------------------- #

class BaselineFusionModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, video_dim=1000, hidden_dim=512, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(text_dim + audio_dim + video_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, audio, video):
        x = torch.cat((text, audio, video), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


# ----------------------------- #
#       TRAINING FUNCTION       #
# ----------------------------- #

def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for text, audio, video, labels in loader:
        text, audio, video, labels = text.to(DEVICE), audio.to(DEVICE), video.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(text, audio, video)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for text, audio, video, labels in loader:
            text, audio, video, labels = text.to(DEVICE), audio.to(DEVICE), video.to(DEVICE), labels.to(DEVICE)
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# ----------------------------- #
#             MAIN              #
# ----------------------------- #

if __name__ == "__main__":
    dataset = MultimodalDataset(MANIFEST, DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BaselineFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, train_accs = [], []

    print(f"🚀 Training on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        loss, acc = train_model(model, loader, criterion, optimizer)
        train_losses.append(loss)
        train_accs.append(acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "baseline_fusion.pt"))
    print("✅ Model saved at results/baseline_fusion.pt")

    # Plot Accuracy
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in train_accs], marker='o')
    plt.title("Baseline Fusion Model Accuracy Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "baseline_accuracy_curve.png"))
    plt.show()

    print(f"📊 Final Accuracy: {train_accs[-1]*100:.2f}%")
