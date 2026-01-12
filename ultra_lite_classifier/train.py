import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, Dataset

class JsonDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        # 2026 Lite Strategy: Keep only high-impact keywords
        self.vocab = {"<UNK>": 0, "look": 1, "buy": 2, "pay": 3, "credit": 4, 
                      "third": 5, "summer": 6, "t-shirt": 7, "card": 8}

    def tokenize(self, text):
        return [self.vocab.get(w.lower(), 0) for w in text.split()]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return torch.tensor(self.tokenize(item['text']), dtype=torch.long), \
               torch.tensor(item['label'], dtype=torch.long)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(_text)
        offsets.append(_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list, offsets, label_list

def train_model(model, dataset, epochs=10, lr=0.01):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for text, offsets, labels in dataloader:
            optimizer.zero_grad()
            output = model(text, offsets)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
