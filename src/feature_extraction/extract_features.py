import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import moviepy.editor as mp

# ---------------- CONFIG ----------------
DATA_DIR = "data/custom"
FEATURE_DIR = "data/features"
os.makedirs(FEATURE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")

# ---------------- TEXT ----------------
def extract_text_features(manifest_path):
    print("📝 Extracting text features...")
    os.makedirs(os.path.join(FEATURE_DIR, "text"), exist_ok=True)
    df = pd.read_csv(manifest_path)
    
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModel.from_pretrained("distilroberta-base").to(DEVICE)
    model.eval()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = open(row["text_path"]).read()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu()
        torch.save(emb, os.path.join(FEATURE_DIR, "text", f"{row['id']}.pt"))

# ---------------- AUDIO ----------------
def extract_audio_features(manifest_path):
    print("🔊 Extracting audio features...")
    os.makedirs(os.path.join(FEATURE_DIR, "audio"), exist_ok=True)
    df = pd.read_csv(manifest_path)

    import soundfile as sf
    import librosa  # we'll use librosa for resampling

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    model.eval()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["audio_path"]
        # Load and resample to 16 kHz
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu()
        torch.save(emb, os.path.join(FEATURE_DIR, "audio", f"{row['id']}.pt"))


# ---------------- VISUAL ----------------
def extract_visual_features(manifest_path):
    print("🎥 Extracting visual features...")
    os.makedirs(os.path.join(FEATURE_DIR, "video"), exist_ok=True)
    df = pd.read_csv(manifest_path)
    
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    resnet.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        clip = mp.VideoFileClip(row["video_path"])
        frame = clip.get_frame(clip.duration / 2)
        img = Image.fromarray(frame)
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = resnet(img_tensor).squeeze().cpu()
        torch.save(emb, os.path.join(FEATURE_DIR, "video", f"{row['id']}.pt"))

# ---------------- RUN ALL ----------------
if __name__ == "__main__":
    manifest = os.path.join(DATA_DIR, "manifest_train.csv")
    extract_text_features(manifest)
    extract_audio_features(manifest)
    extract_visual_features(manifest)
    print("✅ All features extracted and saved to data/features/")
