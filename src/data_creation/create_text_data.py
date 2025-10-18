"""
create_text_data.py
Generates small synthetic text samples for multimodal sentiment dataset.
Each sample = a short sentence + sentiment label (0 = Negative, 1 = Neutral, 2 = Positive)
"""

import os
import random
import csv
from tqdm import tqdm

# ✅ Output directories
OUTPUT_DIR = "data/custom/text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Number of samples to create
NUM_SAMPLES = 15

# ✅ Example text pools
positive_texts = [
    "This product is amazing!",
    "I absolutely love this!",
    "Great experience overall.",
    "Everything worked perfectly.",
    "Super satisfied with the results."
]

neutral_texts = [
    "The product arrived yesterday.",
    "It was okay, nothing special.",
    "Service was fine, as expected.",
    "Delivery time was average.",
    "The color was as shown."
]

negative_texts = [
    "This is terrible.",
    "I really hate how it turned out.",
    "Worst experience ever.",
    "Not worth the money.",
    "It stopped working after a day."
]

sentiment_map = {0: negative_texts, 1: neutral_texts, 2: positive_texts}

# ✅ Generate and save text samples
manifest_path = os.path.join(OUTPUT_DIR, "text_manifest.csv")
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text_path", "label"])
    for i in tqdm(range(NUM_SAMPLES), desc="Generating text samples"):
        label = random.choice([0, 1, 2])
        text = random.choice(sentiment_map[label])
        file_name = f"sample_{i:03d}.txt"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        with open(file_path, "w", encoding="utf-8") as tf:
            tf.write(text)
        writer.writerow([file_path, label])

print(f"✅ Generated {NUM_SAMPLES} text files in {OUTPUT_DIR}")
print(f"📝 Manifest saved to {manifest_path}")
