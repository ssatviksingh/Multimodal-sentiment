"""
create_video_data.py
Generates short synthetic videos representing sentiment categories.
Uses colored backgrounds and text overlays for each label.
0 = Negative (red), 1 = Neutral (gray), 2 = Positive (green)
"""

import os
import random
from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
from tqdm import tqdm

# ✅ Directories
OUTPUT_DIR = "data/custom/video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Config
NUM_SAMPLES = 15
DURATION = 2.5  # seconds
FPS = 24
RESOLUTION = (320, 240)

# ✅ Sentiment label mapping
sentiment_map = {
    0: ("Negative", (255, 0, 0)),      # Red
    1: ("Neutral", (128, 128, 128)),   # Gray
    2: ("Positive", (0, 200, 0))       # Green
}

# ✅ Video generation
for i in tqdm(range(NUM_SAMPLES), desc="Generating video samples"):
    label = random.choice(list(sentiment_map.keys()))
    sentiment_text, color = sentiment_map[label]

    # 🎨 Create background
    bg_clip = ColorClip(size=RESOLUTION, color=color, duration=DURATION)

    # 📝 Add text overlay
    txt_clip = TextClip(
        txt=sentiment_text,
        fontsize=40,
        color='white',
        font='Arial-Bold',
        method='caption',
        size=RESOLUTION
    ).set_position('center').set_duration(DURATION)

    video = CompositeVideoClip([bg_clip, txt_clip])
    out_path = os.path.join(OUTPUT_DIR, f"sample_{i:03d}.mp4")
    video.write_videofile(out_path, fps=FPS, codec='libx264', audio=False, verbose=False, logger=None)
    video.close()

print(f"✅ Generated {NUM_SAMPLES} video files in {OUTPUT_DIR}")
