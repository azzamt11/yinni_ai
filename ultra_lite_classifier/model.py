import torch
import torch.nn as nn

class UltraLiteClassifier(nn.Module):
    def __init__(self, vocab_size=500, num_class=3):
        super.__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 500, sparse=False, mode='mean')
        self.fc = nn.Linear(2, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)