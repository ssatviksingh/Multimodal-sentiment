"""
extract_pretrained_embeddings.py
Generates text/audio/video embeddings using pretrained models:
- Text: DistilBERT
- Audio: Wav2Vec2
- Video: ViT-B/16 (Vision Transformer)
Saves feature tensors (.pt) to data/features/{modality}/ for training.
"""

import os
import cv2
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel,
    Wav2Vec2Processor, Wav2Vec2Model,
    AutoImageProcessor, ViTModel
)
from PIL import Image

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")

MANIFESTS = [
    "data/manifest_train.csv",
    "data/manifest_val.csv",
    "data/manifest_test.csv",
]

OUT_DIR = "data/features"
for sub in ["text", "audio", "video"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

# ---------- LOAD MODELS ----------
print("📚 Loading pretrained models...")

# Text
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE).eval()

# Audio
audio_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()

# Video
image_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()


# ---------- EXTRACTION LOOP ----------
for manifest_path in MANIFESTS:
    print(f"\n🔍 Processing {manifest_path}...")
    df = pd.read_csv(manifest_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        sid = str(row["filename"])
        text = str(row.get("text", ""))
        audio = str(row.get("audio_path", ""))
        video = str(row.get("video_path", ""))

        # skip if already extracted
        if all(os.path.exists(os.path.join(OUT_DIR, m, f"{sid}.pt")) for m in ["text", "audio", "video"]):
            continue

        # ---------- TEXT ----------
        try:
            with torch.no_grad():
                inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
                text_feat = text_model(**inputs).last_hidden_state.mean(dim=1).cpu()
            torch.save(text_feat, os.path.join(OUT_DIR, "text", f"{sid}.pt"))
        except Exception as e:
            print(f"⚠️ Text error for {sid}: {e}")

        # ---------- AUDIO ----------
        try:
            if os.path.exists(audio):
                wav, sr = torchaudio.load(audio)
                inputs = audio_proc(wav.squeeze(0), sampling_rate=sr, return_tensors="pt").input_values.to(DEVICE)
                with torch.no_grad():
                    audio_feat = audio_model(inputs).last_hidden_state.mean(dim=1).cpu()
                torch.save(audio_feat, os.path.join(OUT_DIR, "audio", f"{sid}.pt"))
            else:
                print(f"⚠️ Missing audio file: {audio}")
        except Exception as e:
            print(f"⚠️ Audio error for {sid}: {e}")

        # ---------- VIDEO ----------
                # ---------- VIDEO ----------
        try:
            if os.path.exists(video):
                cap = cv2.VideoCapture(video)
                success, frame = cap.read()
                cap.release()
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inputs = image_proc(images=Image.fromarray(frame_rgb), return_tensors="pt")["pixel_values"].to(DEVICE)
                    with torch.no_grad():
                        video_feat = vit_model(inputs).last_hidden_state.mean(dim=1).cpu()
                    torch.save(video_feat, os.path.join(OUT_DIR, "video", f"{sid}.pt"))
                else:
                    print(f"⚠️ Could not read frame from video: {video}")
            else:
                print(f"⚠️ Missing video file: {video}")
        except Exception as e:
            print(f"⚠️ Video error for {sid}: {e}")


print("✅ Saved all pretrained embeddings to data/features/")
