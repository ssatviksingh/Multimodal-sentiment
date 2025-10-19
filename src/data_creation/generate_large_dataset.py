"""
generate_large_dataset.py
──────────────────────────
Generates a scalable synthetic multimodal dataset (text, audio, video)
with the same format as your existing dataset.

💾 Output:
- data/raw/audio/*.wav
- data/raw/video/*.mp4
- data/manifest_train.csv
- data/manifest_val.csv
- data/manifest_test.csv

✅ Usage:
python -m src.data_creation.generate_large_dataset --samples 100000 --duration 0.5
"""

import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import wave
import struct
import cv2
import multiprocessing as mp

# ---------- CONFIG ----------
TEXT_SETS = {
    0: [
        "This was terrible, I hated every moment.",
        "Awful experience, not worth watching.",
        "Completely disappointed, waste of time."
    ],
    1: [
        "It was okay, not great but not bad either.",
        "An average performance, could be better.",
        "Mediocre storyline but decent acting."
    ],
    2: [
        "Absolutely loved it, fantastic experience!",
        "Great film, emotional and powerful.",
        "Wonderful, exceeded all expectations!"
    ]
}

AUDIO_DIR = "data/raw/audio"
VIDEO_DIR = "data/raw/video"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
MANIFEST_DIR = "data"
os.makedirs(MANIFEST_DIR, exist_ok=True)

# ---------- GENERATORS ----------
def generate_sine_wave(filename, duration=0.5, freq=440.0, sample_rate=16000):
    """Generate a simple sine wave .wav audio file"""
    samples = int(sample_rate * duration)
    t = np.arange(samples)
    wave_data = np.sin(2 * np.pi * freq * t / sample_rate)
    scaled = np.int16(wave_data * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack('<' + ('h' * len(scaled)), *scaled))

def generate_dummy_video(filename, color=(0, 0, 255), duration=0.5, fps=12):
    """Generate a simple solid-color video"""
    width, height = 224, 224
    frames = int(fps * duration)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(frames):
        out.write(frame)
    out.release()

# ---------- WORKER ----------
def create_sample(i):
    label = random.choice([0, 1, 2])
    text = random.choice(TEXT_SETS[label])
    filename = f"sample_{i:06d}"
    audio_path = os.path.join(AUDIO_DIR, f"{filename}.wav")
    video_path = os.path.join(VIDEO_DIR, f"{filename}.mp4")

    freq = 300 + (label * 200)
    color = [(255, 0, 0), (255, 255, 0), (0, 255, 0)][label]
    generate_sine_wave(audio_path, freq=freq)
    generate_dummy_video(video_path, color=color)

    return {
        "filename": filename,
        "audio_path": audio_path,
        "video_path": video_path,
        "text": text,
        "label": label
    }

# ---------- MAIN ----------
def generate_dataset(num_samples, workers=8):
    print(f"🎬 Generating {num_samples:,} synthetic multimodal samples using {workers} workers...")
    with mp.Pool(processes=workers) as pool:
        rows = list(tqdm(pool.imap(create_sample, range(num_samples)), total=num_samples))

    df = pd.DataFrame(rows)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
    test_df = df.drop(train_df.index).drop(val_df.index)

    train_df.to_csv(os.path.join(MANIFEST_DIR, "manifest_train.csv"), index=False)
    val_df.to_csv(os.path.join(MANIFEST_DIR, "manifest_val.csv"), index=False)
    test_df.to_csv(os.path.join(MANIFEST_DIR, "manifest_test.csv"), index=False)

    print(f"✅ Dataset generated successfully!")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------- ENTRY ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate (e.g., 100000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--duration", type=float, default=0.5, help="Audio/video duration in seconds")
    args = parser.parse_args()

    generate_dataset(args.samples, args.workers)
