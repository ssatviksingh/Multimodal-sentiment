"""
extract_pretrained_embeddings.py
Generates text/audio/video embeddings using pretrained models:
- Text: DistilBERT
- Audio: Wav2Vec2
- Video: ViT-B/16 (Vision Transformer)
Saves feature tensors (.pt) to data/features/{modality}/ for training.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel,
    Wav2Vec2Processor, Wav2Vec2Model,
    AutoImageProcessor, ViTModel
)
import librosa
import cv2
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")

# ---------- PATHS ----------
MANIFEST = "data/custom/manifest_train.csv"
OUT_BASE = "data/features"
os.makedirs(OUT_BASE, exist_ok=True)

os.makedirs(f"{OUT_BASE}/text", exist_ok=True)
os.makedirs(f"{OUT_BASE}/audio", exist_ok=True)
os.makedirs(f"{OUT_BASE}/video", exist_ok=True)

# ---------- LOAD MODELS ----------
print("📚 Loading pretrained models...")

# Text
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

# Audio
audio_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

# Video (ViT)
image_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)

# ---------- EXTRACTORS ----------

def extract_text_feat(text):
    tokens = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        out = text_model(**tokens).last_hidden_state.mean(dim=1)
    return out.squeeze(0).cpu()

def extract_audio_feat(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    inputs = audio_proc(y, sampling_rate=sr, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = audio_model(**inputs).last_hidden_state.mean(dim=1)
    return emb.squeeze(0).cpu()

def extract_video_feat(path, num_frames=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return torch.zeros(768)
    idxs = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        inputs = image_proc(frame, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = vit_model(**inputs).pooler_output
        frames.append(out.cpu())
    cap.release()
    if not frames:
        return torch.zeros(768)
    return torch.stack(frames).mean(dim=0).squeeze(0)

# ---------- MAIN ----------
df = pd.read_csv(MANIFEST)
df.columns = [c.strip().lower() for c in df.columns]

print(f"🧩 Extracting features for {len(df)} samples...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    sid = str(row.get("filename") or row.get("id") or f"sample_{i:03d}")
    text = row.get("text") or row.get("sentence") or row.get("utterance") or ""
    label = int(row["label"])

    text_feat = extract_text_feat(text)
    torch.save(text_feat, f"{OUT_BASE}/text/{sid}.pt")

    audio_feat = extract_audio_feat(row["audio_path"])
    torch.save(audio_feat, f"{OUT_BASE}/audio/{sid}.pt")

    video_feat = extract_video_feat(row["video_path"])
    torch.save(video_feat, f"{OUT_BASE}/video/{sid}.pt")

print("✅ Saved all pretrained embeddings to data/features/")
