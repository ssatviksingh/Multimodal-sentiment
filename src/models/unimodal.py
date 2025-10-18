
# unimodal.py - wrappers for text/audio/vision encoders using Hugging Face / torchvision
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, Wav2Vec2Model
import torchvision.models as models

class TextEncoder(nn.Module):
    def __init__(self, model_name='roberta-base', out_dim=768, freeze=True):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.cfg)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.out_dim = out_dim

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # use pooled output or mean of last hidden
        pooled = out.last_hidden_state.mean(dim=1)
        return pooled  # (B, hidden)

class AudioEncoder(nn.Module):
    def __init__(self, model_name='facebook/wav2vec2-base-960h', freeze=True):
        super().__init__()
        # Using wav2vec2 to extract features from raw waveform is ideal.
        # Here we assume we already extracted MFCCs; for full wav2vec pipeline you'd pass raw audio.
        # We'll implement a tiny CNN over MFCC features for example.
        self.cnn = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.out_dim = 64

    def forward(self, mfcc):
        # mfcc: (B, T, F) => convert to (B, F, T)
        x = mfcc.permute(0,2,1)
        return self.cnn(x)

class VisionEncoder(nn.Module):
    def __init__(self, out_dim=512, freeze=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove classifier
        self.backbone = nn.Sequential(*modules)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.out_dim = out_dim

    def forward(self, frames):
        # frames: (B, num_frames, H, W, C)
        B, F, H, W, C = frames.shape
        # flatten frames and pass through backbone
        x = frames.view(B*F, H, W, C).permute(0,3,1,2).float() / 255.0
        feat = self.backbone(x).view(B, F, -1)  # (B, F, feat)
        pooled = feat.mean(dim=1)
        return pooled

