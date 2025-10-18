
# train.py - minimal training loop
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import MultimodalDataset
from src.models.fusion_transformer import FusionModel
from src.utils import set_seed, ensure_dir
from tqdm import tqdm
import numpy as np

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="train"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        label = batch['label'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn, audio=audio, vision=vision)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            label = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn, audio=audio, vision=vision)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(label.cpu().numpy().tolist())
    acc = (np.array(preds) == np.array(labels)).mean()
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed", help="processed data root")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = MultimodalDataset(root_dir=args.data_dir, split='train', tokenizer=tokenizer)
    val_ds = MultimodalDataset(root_dir=args.data_dir, split='val', tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FusionModel(text_model=args.model_name, num_classes=args.num_classes, freeze_encoders=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ensure_dir(args.out_dir)
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print("Train Loss:", train_loss)
        val_acc = eval_epoch(model, val_loader, device)
        print("Val Acc:", val_acc)
        ckpt = os.path.join(args.out_dir, f"model_epoch{epoch+1}.pt")
        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'acc': val_acc}, ckpt)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

if __name__ == "__main__":
    main()

