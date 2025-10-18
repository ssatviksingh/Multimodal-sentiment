
# evaluate.py - load best checkpoint and compute metrics
import argparse
import torch
from transformers import AutoTokenizer
from src.dataset import MultimodalDataset
from src.models.fusion_transformer import FusionModel
from src.utils import set_seed
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--num_classes", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_ds = MultimodalDataset(root_dir=args.data_dir, split='test', tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=8)
    model = FusionModel(text_model=args.model_name, num_classes=args.num_classes, freeze_encoders=True)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            label = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn, audio=audio, vision=vision)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(label.cpu().numpy().tolist())

    print(classification_report(labels, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()

