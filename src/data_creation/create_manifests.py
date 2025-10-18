import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define directories
BASE_DIR = "data/custom"
TEXT_DIR = os.path.join(BASE_DIR, "text")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
VIDEO_DIR = os.path.join(BASE_DIR, "video")

# Collect sample names (assuming same filenames across all modalities)
samples = sorted([f.replace(".txt", "") for f in os.listdir(TEXT_DIR) if f.endswith(".txt")])

data = []
for s in samples:
    text_path = os.path.join(TEXT_DIR, f"{s}.txt")
    audio_path = os.path.join(AUDIO_DIR, f"{s}.wav")
    video_path = os.path.join(VIDEO_DIR, f"{s}.mp4")

    # Random label for now: Negative(0), Neutral(1), Positive(2)
    label = hash(s) % 3
    data.append({
        "id": s,
        "text_path": text_path,
        "audio_path": audio_path,
        "video_path": video_path,
        "label": label
    })

df = pd.DataFrame(data)

# Split into train/val/test
train, temp = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

# Save CSVs
os.makedirs(BASE_DIR, exist_ok=True)
train.to_csv(os.path.join(BASE_DIR, "manifest_train.csv"), index=False)
val.to_csv(os.path.join(BASE_DIR, "manifest_val.csv"), index=False)
test.to_csv(os.path.join(BASE_DIR, "manifest_test.csv"), index=False)

print(f"✅ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print("📂 Manifests saved in data/custom/")
