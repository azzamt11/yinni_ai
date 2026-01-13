import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class JsonDataset(Dataset):
    def __init__(self, json_path, vocab=None, max_length=None):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        
        self.vocab = vocab or self.build_vocab()
        self.max_length = max_length
        self.vocab_size = len(self.vocab)
    
    def build_vocab(self):
        vocab = {'<unk>': 0, '<pad>': 1}
        for item in self.samples:
            words = item['text'].lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def tokenize(self, text):
        tokens = [self.vocab.get(w.lower(), 0) for w in text.split()]
        if self.max_length:
            tokens = tokens[:self.max_length]
        return tokens
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenize(item['text'])
        return torch.tensor(tokens, dtype=torch.long), \
               torch.tensor(item['label'], dtype=torch.long)

def collate_batch(batch):
    texts, labels = zip(*batch)
    # Pad sequences
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=1)
    labels = torch.stack(labels)
    return padded_texts, labels

# Simple model example
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embeds = self.embedding(x)
        _, (hidden, _) = self.rnn(embeds)
        return self.fc(hidden[-1])

def train_model(model, dataset, epochs=7, lr=0.001):
    # Create collate function for EmbeddingBag
    def collate_batch_embeddingbag(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(_label)
            text_list.append(_text)
            offsets.append(_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, offsets, label_list
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, 
                          collate_fn=collate_batch_embeddingbag)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss, total_correct = 0, 0
        for text, offsets, labels in dataloader:
            optimizer.zero_grad()
            output = model(text, offsets)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()
        
        acc = total_correct / len(dataset)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")