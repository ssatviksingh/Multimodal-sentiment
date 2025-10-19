"""
extract_features.py
Extracts low-level features for each modality:
 - Audio → MFCC
 - Video → average color histogram
 - Text → raw text (optionally pretrained embeddings)
"""

import os
import csv
import librosa
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

DATA_ROOT = "data"
FEATURE_DIR = os.path.join(DATA_ROOT, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

AUDIO_FEAT_DIR = os.path.join(FEATURE_DIR, "audio")
VIDEO_FEAT_DIR = os.path.join(FEATURE_DIR, "video")
TEXT_FEAT_DIR = os.path.join(FEATURE_DIR, "text")
for d in [AUDIO_FEAT_DIR, VIDEO_FEAT_DIR, TEXT_FEAT_DIR]:
    os.makedirs(d, exist_ok=True)

# Optional: load pretrained model for text
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base").eval()

def extract_audio_features(path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"[AudioError] {path}: {e}")
        return np.zeros(n_mfcc)

def extract_video_features(path, num_frames=8):
    try:
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // num_frames)
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            hist = cv2.calcHist([frame], [0,1,2], None, [8,8,8],
                                [0,256,0,256,0,256]).flatten()
            frames.append(hist)
        cap.release()
        return np.mean(frames, axis=0) if frames else np.zeros(512)
    except Exception as e:
        print(f"[VideoError] {path}: {e}")
        return np.zeros(512)

def extract_text_features(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Process all manifests
for split in ["train", "val", "test"]:
    manifest_path = os.path.join(DATA_ROOT, f"manifest_{split}.csv")
    if not os.path.exists(manifest_path):
        continue
    print(f"\n🔍 Processing {manifest_path}...")
    features = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            filename = row["filename"]
            audio_path = row["audio_path"]
            video_path = row["video_path"]
            text = row["text"]
            label = int(row["label"])

            a_feat = extract_audio_features(audio_path)
            v_feat = extract_video_features(video_path)
            t_feat = extract_text_features(text)

            out = {
                "filename": filename,
                "audio_feat": a_feat,
                "video_feat": v_feat,
                "text_feat": t_feat,
                "label": label
            }
            features.append(out)

    # Save features as .pkl
    out_path = os.path.join(FEATURE_DIR, f"{split}_features.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(features, f)
    print(f"✅ Saved {len(features)} feature samples to {out_path}")
