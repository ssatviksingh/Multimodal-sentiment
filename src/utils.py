# utils.py - helper utilities
import os
import random
import numpy as np
import torch
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path) as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

