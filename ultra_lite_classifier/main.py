import os
from model import UltraLiteClassifier
from train import JsonDataset, train_model
import json
import pickle
import torch

def save_training_artifacts(model, dataset, class_names=['Find_Item', 'Select_Option', 'Make_Payment']):
    """
    Save model artifacts for inference
    """
    # Get ACTUAL vocabulary size from dataset
    actual_vocab_size = dataset.vocab_size
    
    print(f"\nSaving model artifacts...")
    print(f"   Actual vocab size: {actual_vocab_size}")
    print(f"   Number of classes: {len(class_names)}")
    
    # Save model weights
    torch.save(model.state_dict(), 'ulc_model_weights.pth')
    print(f"Saved: ulc_model_weights.pth")
    
    # Save vocabulary
    with open('ulc_vocab.pkl', 'wb') as f:
        pickle.dump(dataset.vocab, f)
    print(f"Saved: ulc_vocab.pkl ({len(dataset.vocab)} words)")
    
    # Save configuration with CORRECT vocab size
    config = {
        'vocab_size': actual_vocab_size,
        'num_classes': len(class_names),
        'embed_dim': 128,
        'class_names': class_names
    }
    
    with open('ulc_model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved: ulc_model_config.json")
    
    print(f"\nModel artifacts saved for inference!")
    print(f"   Run 'python test.py' to test the model")

def main():
    # Setup paths for 2026 environment
    base_path = os.path.dirname(__file__)
    json_path = os.path.join(base_path, 'dataset.json')
    
    # Define class names based on your dataset
    class_names = ['Find_Item', 'Select_Option', 'Make_Payment']

    # 1. Initialize Dataset
    print("Loading dataset...")
    dataset = JsonDataset(json_path)
    print(f"   Loaded {len(dataset)} samples")
    print(f"   Vocabulary size: {dataset.vocab_size}")

    # 2. Initialize Model with CORRECT vocabulary size
    print("\nInitializing model...")
    model = UltraLiteClassifier(
        vocab_size=dataset.vocab_size,  # Use actual vocab size
        num_class=len(class_names)
    )
    
    # 3. Parameter count check
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {params:,}")

    # 4. Start Training
    print("\nStarting training...")
    train_model(model, dataset, epochs=7)
    
    # 5. Save artifacts for inference
    save_training_artifacts(model, dataset, class_names)

if __name__ == "__main__":
    main()