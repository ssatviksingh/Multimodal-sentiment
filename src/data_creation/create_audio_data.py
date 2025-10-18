"""
create_audio_data.py
Generates audio samples using text-to-speech (gTTS) for each text entry.
"""
from gtts import gTTS
import pandas as pd, os
from tqdm import tqdm
from pydub import AudioSegment
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

df = pd.read_csv("data/custom/text_data.csv")
os.makedirs("data/custom/audio", exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    fname = row["filename"]
    text = row["text"]
    audio_path = f"data/custom/audio/{fname}.wav"
    try:
        tts = gTTS(text=text, lang="en")
        temp_mp3 = f"{audio_path.replace('.wav','.mp3')}"
        tts.save(temp_mp3)
        sound = AudioSegment.from_mp3(temp_mp3)
        sound.export(audio_path, format="wav")
        os.remove(temp_mp3)
    except Exception as e:
        print(f"⚠️ Error on {fname}: {e}")

print("✅ Audio generated in data/custom/audio/")
