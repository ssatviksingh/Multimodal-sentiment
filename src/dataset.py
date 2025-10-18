
# dataset.py - minimal dataset skeleton (adapt to dataset specifics)
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """
    Expect processed data folder with npz files per example:
    {'text': '<string>', 'audio': np.array(...) , 'vision': np.array(...), 'label': int}
    This is a minimal example that loads cached features.
    """
    def __init__(self, root_dir, split='train', tokenizer=None, max_text_len=128):
        super().__init__()
        self.root = os.path.join(root_dir, split)
        self.files = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.npz')]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        text = str(data['text'].item())
        audio = data['audio'].astype(np.float32)
        vision = data['vision'].astype(np.float32)
        label = int(data['label'].item())

        # Tokenize text to input ids if tokenizer provided
        if self.tokenizer:
            enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_text_len, return_tensors='pt')
            input_ids = enc['input_ids'].squeeze(0)
            attn = enc['attention_mask'].squeeze(0)
        else:
            # fallback: represent text as zeros
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attn = torch.zeros(self.max_text_len, dtype=torch.long)

        # collapse or pad audio/vision features to fixed shapes (simple)
        audio_tensor = torch.from_numpy(audio)
        vision_tensor = torch.from_numpy(vision)

        return {
            'input_ids': input_ids,
            'attention_mask': attn,
            'audio': audio_tensor,
            'vision': vision_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

