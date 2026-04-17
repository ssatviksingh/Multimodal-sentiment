"""
extract_pretrained_embeddings.py
Generates text/audio/video embeddings using pretrained models:
- Text: DistilBERT
- Audio: Wav2Vec2 (16 kHz mono, length-capped)
- Video: ViT-B/16 — multiple frames mean-pooled to 768-d
Saves feature tensors (.pt) to OUT_DIR/{text,audio,video}/

Usage:
  python src/feature_extraction/pretrained/extract_pretrained_embeddings.py
  python src/feature_extraction/pretrained/extract_pretrained_embeddings.py --out-dir data/features_v2 --video-frames 8
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    AutoImageProcessor,
    ViTModel,
)
from PIL import Image

TEXT_MAX_LENGTH = 256
MAX_AUDIO_SAMPLES_16K = 320000  # ~20 s at 16 kHz mono


def parse_args():
    p = argparse.ArgumentParser(description="Extract multimodal embeddings into OUT_DIR/{text,audio,video}/")
    p.add_argument("--out-dir", default="data/features", help="Output root (e.g. data/features_v2)")
    p.add_argument("--video-frames", type=int, default=8, help="Frames to sample per video (evenly spaced)")
    p.add_argument(
        "--manifests",
        nargs="*",
        default=["data/manifest_train.csv", "data/manifest_val.csv", "data/manifest_test.csv"],
        help="Manifest CSV paths",
    )
    return p.parse_args()


def sample_frame_indices(n_frames: int, k: int) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    k = min(k, n_frames)
    return np.unique(np.linspace(0, n_frames - 1, num=k, dtype=float).astype(int))


def extract_video_embedding(
    video_path: str,
    vit_model: ViTModel,
    image_proc,
    device: str,
    num_frames: int,
) -> torch.Tensor | None:
    """Mean-pool ViT patch embeddings across sampled frames -> [1, 768]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_frames <= 0:
        cap.release()
        return None
    idxs = sample_frame_indices(n_frames, num_frames)
    frame_embs: list[torch.Tensor] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = image_proc(images=Image.fromarray(frame_rgb), return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            out = vit_model(inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu()
        frame_embs.append(out)
    cap.release()
    if not frame_embs:
        return None
    stacked = torch.stack(frame_embs, dim=0)
    return stacked.mean(dim=0, keepdim=True)


def prepare_audio_waveform(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    """Mono, resample to 16 kHz, center-crop if too long."""
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    n = wav.shape[-1]
    if n > MAX_AUDIO_SAMPLES_16K:
        start = (n - MAX_AUDIO_SAMPLES_16K) // 2
        wav = wav[..., start : start + MAX_AUDIO_SAMPLES_16K]
    return wav, sr


def run_extraction(
    out_dir: str,
    video_frames: int,
    manifest_paths: list[str],
    device: str,
):
    for sub in ["text", "audio", "video"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    print("📚 Loading pretrained models...")
    text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

    audio_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    image_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()

    for manifest_path in manifest_paths:
        if not os.path.isfile(manifest_path):
            print(f"⚠️ Skip missing manifest: {manifest_path}")
            continue
        print(f"\n🔍 Processing {manifest_path}...")
        df = pd.read_csv(manifest_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            sid = str(row["filename"])
            text = str(row.get("text", ""))
            audio_path = str(row.get("audio_path", ""))
            video_path = str(row.get("video_path", ""))

            if all(os.path.exists(os.path.join(out_dir, m, f"{sid}.pt")) for m in ["text", "audio", "video"]):
                continue

            # ---------- TEXT ----------
            try:
                with torch.no_grad():
                    inputs = text_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=TEXT_MAX_LENGTH,
                    ).to(device)
                    text_feat = text_model(**inputs).last_hidden_state.mean(dim=1).cpu()
                torch.save(text_feat, os.path.join(out_dir, "text", f"{sid}.pt"))
            except Exception as e:
                print(f"⚠️ Text error for {sid}: {e}")

            # ---------- AUDIO ----------
            try:
                if os.path.exists(audio_path):
                    wav, sr = torchaudio.load(audio_path)
                    wav, sr = prepare_audio_waveform(wav, sr)
                    inputs = audio_proc(wav.squeeze(0), sampling_rate=sr, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        audio_feat = audio_model(inputs).last_hidden_state.mean(dim=1).cpu()
                    torch.save(audio_feat, os.path.join(out_dir, "audio", f"{sid}.pt"))
                else:
                    print(f"⚠️ Missing audio file: {audio_path}")
            except Exception as e:
                print(f"⚠️ Audio error for {sid}: {e}")

            # ---------- VIDEO (multi-frame mean pool) ----------
            try:
                if os.path.exists(video_path):
                    video_feat = extract_video_embedding(
                        video_path, vit_model, image_proc, device, video_frames
                    )
                    if video_feat is not None:
                        torch.save(video_feat, os.path.join(out_dir, "video", f"{sid}.pt"))
                    else:
                        print(f"⚠️ No video frames for {sid}: {video_path}")
                else:
                    print(f"⚠️ Missing video file: {video_path}")
            except Exception as e:
                print(f"⚠️ Video error for {sid}: {e}")

    print(f"✅ Saved pretrained embeddings under {out_dir}/")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device} | out-dir={args.out_dir} | video-frames={args.video_frames}")
    run_extraction(args.out_dir, args.video_frames, args.manifests, device)


if __name__ == "__main__":
    main()
