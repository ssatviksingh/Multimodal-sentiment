"""
generate_dummy_dataset.py
Creates a large synthetic multimodal dataset:
 - Random short audio (sine + noise)
 - Solid-color video clips
 - Text sentences (positive, neutral, negative)
 - Generates train/val/test manifest CSVs
"""

import os, csv, random, numpy as np, cv2, soundfile as sf
from tqdm import tqdm

DATA_ROOT = "data/raw"
os.makedirs(f"{DATA_ROOT}/audio", exist_ok=True)
os.makedirs(f"{DATA_ROOT}/video", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Sentiment categories
sentiments = {
    2: [  # Positive
        "I absolutely loved this!",
        "That was wonderful.",
        "Amazing experience overall.",
        "I’m so happy with the result!",
        "Great visuals and sound quality!"
    ],
    1: [  # Neutral
        "It was okay, nothing special.",
        "Average experience overall.",
        "Not bad, but not great either.",
        "Just fine for a one-time watch.",
        "Decent but forgettable."
    ],
    0: [  # Negative
        "I didn’t like it at all.",
        "Terrible and disappointing.",
        "Poor performance and bad story.",
        "Completely boring experience.",
        "I would not recommend this."
    ]
}

def make_audio(path, duration=2.0, sr=16000, freq=440):
    """Create random sine-wave + noise audio."""
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.1 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(len(t))
    sf.write(path, y, sr)

def make_video(path, color=(0, 255, 0), frames=16, size=(224, 224)):
    """Create solid-color video clip."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 8.0, size)
    for _ in range(frames):
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()

# 🧮 Generate larger dataset
TOTAL_SAMPLES = 300  # change to 500+ if you want
splits = {"train": 0.7, "val": 0.15, "test": 0.15}

# Prepare dataset
samples = []
for i in tqdm(range(TOTAL_SAMPLES), desc="Generating samples"):
    label = random.choice([0, 1, 2])
    text = random.choice(sentiments[label])
    fname = f"sample_{i:04d}"
    a_path = f"{DATA_ROOT}/audio/{fname}.wav"
    v_path = f"{DATA_ROOT}/video/{fname}.mp4"

    # Random audio pitch & color
    freq = random.randint(300, 700)
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    make_audio(a_path, freq=freq)
    make_video(v_path, color=color)

    samples.append({
        "filename": fname,
        "audio_path": a_path,
        "video_path": v_path,
        "text": text,
        "label": label
    })

# Split dataset
random.shuffle(samples)
n_train = int(splits["train"] * len(samples))
n_val = int(splits["val"] * len(samples))
train_split = samples[:n_train]
val_split = samples[n_train:n_train + n_val]
test_split = samples[n_train + n_val:]

# Write manifests
for name, data in zip(["train", "val", "test"], [train_split, val_split, test_split]):
    manifest = f"data/manifest_{name}.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "audio_path", "video_path", "text", "label"])
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Wrote {manifest} with {len(data)} samples")

print("🎯 Synthetic multimodal dataset generated successfully!")
