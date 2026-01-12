import os
from model import UltraLiteClassifier
from train import JsonDataset, train_model

def main():
    # Setup paths for 2026 environment
    base_path = os.path.dirname(__file__)
    json_path = os.path.join(base_path, 'dataset.json')

    # 1. Initialize Dataset
    dataset = JsonDataset(json_path)

    # 2. Initialize Model (20 vocab size, 3 classes)
    model = UltraLiteClassifier(vocab_size=500, num_class=3)
    
    # 3. Parameter count check (< 100 limit)
    params = sum(p.numel() for p in model.parameters())
    print(f"Initializing model with {params} parameters...")

    # 4. Start Training
    train_model(model, dataset, epochs=15)

if __name__ == "__main__":
    main()
