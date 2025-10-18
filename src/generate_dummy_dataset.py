
"""
generate_dummy_dataset.py
Creates a small synthetic multimodal dataset:
 - Generates short random WAV audio
 - Creates solid-color MP4 video clips
 - Writes simple sentiment text samples
 - Builds manifest_train.csv, manifest_val.csv, manifest_test.csv
"""
import os, csv, random, numpy as np, cv2, soundfile as sf

DATA_ROOT = "data/raw"
os.makedirs(f"{DATA_ROOT}/audio", exist_ok=True)
os.makedirs(f"{DATA_ROOT}/video", exist_ok=True)

samples = [
    ("I loved this movie, it was fantastic!", 2),
    ("Terrible acting, I hated it.", 0),
    ("Average film, nothing special.", 1),
    ("Amazing visuals and story!", 2),
    ("Poor sound quality, bad experience.", 0),
    ("It was okay, could be better.", 1),
]

splits = {
    "train": samples[:3],
    "val": samples[3:4],
    "test": samples[4:]
}

def make_audio(path, duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(sr*duration))
    y = 0.1*np.sin(2*np.pi*440*t) + 0.05*np.random.randn(len(t))
    sf.write(path, y, sr)

def make_video(path, color=(0,255,0), frames=16, size=(224,224)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 8.0, size)
    for _ in range(frames):
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()

for split, data in splits.items():
    rows = []
    for i, (text, label) in enumerate(data):
        fname = f"{split}_{i:03d}"
        a_path = f"{DATA_ROOT}/audio/{fname}.wav"
        v_path = f"{DATA_ROOT}/video/{fname}.mp4"
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        make_audio(a_path)
        make_video(v_path, color=color)
        rows.append({
            "filename": fname,
            "audio_path": a_path,
            "video_path": v_path,
            "text": text,
            "label": label
        })
    os.makedirs("data", exist_ok=True)
    manifest = f"data/manifest_{split}.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","audio_path","video_path","text","label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest} with {len(rows)} samples")

print("✅ Dummy dataset generated under data/raw and manifest CSVs created.")

