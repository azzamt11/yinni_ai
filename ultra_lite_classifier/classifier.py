import torch
import torch.nn as nn
import json
import pickle
import os
import sys
from model import UltraLiteClassifier
from rapidfuzz import process, fuzz

class TextPredictor:
    def __init__(self, model_path='ulc_model_weights.pth', 
                 vocab_path='ulc_vocab.pkl', 
                 config_path='ulc_model_config.json'):
        """
        Load trained model for inference
        """
        # Check if files exist
        missing_files = []
        for path in [model_path, vocab_path, config_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            print("Missing files. Please train the model first by running:")
            print("   python main.py")
            print("\nMissing files:")
            for f in missing_files:
                print(f"   - {f}")
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Get ACTUAL vocabulary size from the loaded vocab
        actual_vocab_size = len(self.vocab)
        config_vocab_size = self.config.get('vocab_size', actual_vocab_size)
        
        # Use the larger of the two to be safe
        vocab_size_to_use = max(actual_vocab_size, config_vocab_size)
        
        # print(f"Vocabulary info:")
        # print(f"   - Loaded vocab size: {actual_vocab_size}")
        # print(f"   - Config vocab size: {config_vocab_size}")
        # print(f"   - Using vocab size: {vocab_size_to_use}")
        
        # Initialize model with CORRECT vocabulary size
        self.model = UltraLiteClassifier(
            vocab_size=vocab_size_to_use,
            num_class=self.config['num_classes'],
            embed_dim=self.config.get('embed_dim', 128)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if checkpoint matches model
        model_state_dict = self.model.state_dict()
        checkpoint_keys = set(checkpoint.keys())
        model_keys = set(model_state_dict.keys())
        
        # print(f"\nModel checkpoint analysis:")
        # print(f"   - Checkpoint keys: {checkpoint_keys}")
        # print(f"   - Model keys: {model_keys}")
        
        # Try to load state dict
        try:
            self.model.load_state_dict(checkpoint, strict=True)
            # print("Model loaded successfully!")
        except RuntimeError as e:
            # print(f"Warning: {e}")
            # print("   Trying partial load...")
            
            # Try partial load
            self.model.load_state_dict(checkpoint, strict=False)
            # print("Model loaded (partial match)")
        
        self.model.eval()  # Set to evaluation mode
        
        # Get class names
        self.class_names = self.config.get('class_names', ['Type_0', 'Type_1', 'Type_2'])
        
        # print(f"\nModel ready for inference!")
        # print(f"   Classes: {self.class_names}")
        # print(f"   Embedding dimension: {self.config.get('embed_dim', 128)}")
        # print(f"   Number of classes: {self.config['num_classes']}")
    
    def tokenize_text(self, text):
        """
        Convert text to token indices using the saved vocabulary
        """
        # Same preprocessing as during training
        text = text.lower().strip()
        words = text.split()
        
        tokens = []
        for word in words:
            # Use vocab.get(word, 0) where 0 is <unk> token
            token = self.vocab.get(word, 0)
            if token >= self.model.embedding.num_embeddings:
                token = 0  # Fallback to <unk> if token ID is out of bounds
            tokens.append(token)
        
        if not tokens:  # If empty input
            tokens = [0]  # Use unknown token
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def predict(self, text, return_details=False):
        """
        Predict class for input text
        """
        try:
            # Tokenize
            tokens = self.tokenize_text(text)
            
            # Check if we have any valid tokens
            if len(tokens) == 0:
                return "Unknown" if not return_details else {"error": "No valid tokens"}
            
            # Prepare for EmbeddingBag
            text_tensor = tokens
            offsets = torch.tensor([0])
            
            # Debug info
            # print(f"Tokenized: {len(tokens)} tokens")
            
            # Inference
            with torch.no_grad():
                output = self.model(text_tensor, offsets)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            
            predicted_class = self.class_names[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            if return_details:
                # Get all probabilities
                probs_dict = {}
                for i, class_name in enumerate(self.class_names):
                    probs_dict[class_name] = probabilities[0][i].item()
                
                return {
                    'text': text,
                    'predicted_class': predicted_class,
                    'class_id': predicted_class_idx,
                    'confidence': confidence,
                    'probabilities': probs_dict,
                    'num_tokens': len(tokens)
                }
            else:
                return predicted_class, confidence
                
        except Exception as e:
            print(f"Prediction error: {e}")
            if return_details:
                return {"error": str(e)}
            return "Error", 0.0
    
    def predict_batch(self, texts):
        """
        Predict classes for multiple texts at once
        """
        results = []
        for text in texts:
            try:
                pred_class, confidence = self.predict(text)
                results.append({
                    'text': text,
                    'predicted_class': pred_class,
                    'confidence': confidence
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
        
        return results

def check_model_files():
    """
    Check if model files exist and show information
    """
    files_to_check = [
        ('ulc_model_weights.pth', 'Model weights'),
        ('ulc_vocab.pkl', 'Vocabulary'),
        ('ulc_model_config.json', 'Model config')
    ]
    
    print("🔍 Checking model files...")
    all_exist = True
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"{description}: {filename} ({size:,} bytes)")
        else:
            print(f"{description}: {filename} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\nTo create model files, run:")
        print("   python main.py")
    
    return all_exist

def interactive_test():
    """
    Interactive command-line interface for testing
    """

    # ===== HEADER / BANNERS (MUST NOT APPEAR) =====
    # print("=" * 60)
    # print("TEXT CLASSIFIER - INTERACTIVE MODE")
    # print("=" * 60)

    # First check if files exist
    if not check_model_files():
        # print("\nCannot start. Missing model files.")
        return

    try:
        # Initialize predictor
        predictor = TextPredictor()
    except Exception as e:
        # print(f"Failed to initialize predictor: {e}")
        # print("\nTry re-training the model:")
        # print("   python main.py")
        return

    # ===== COMMAND HELP TEXT (MUST NOT APPEAR) =====
    # print("\nCommands:")
    # print("  - Type any text to classify")
    # print("  - 'batch' : Enter batch mode")
    # print("  - 'prob'  : Toggle probability details")
    # print("  - 'vocab' : Show vocabulary info")
    # print("  - 'model' : Show model info")
    # print("  - 'quit'  : Exit")
    # print("-" * 60)

    show_details = False

    while True:
        try:
            user_input = input("\n📝 Enter text or command: ").strip()

            if user_input.lower() == 'quit':
                # print("Goodbye!")
                break

            elif user_input.lower() == 'prob':
                show_details = not show_details
                # status = "ON" if show_details else "OFF"
                # print(f"Probability details: {status}")
                continue

            elif user_input.lower() == 'vocab':
                # print(f"\nVocabulary:")
                # print(f"   Size: {len(predictor.vocab)}")
                # print(f"   First 10 items:")
                # for i, (word, idx) in enumerate(list(predictor.vocab.items())[:10]):
                #     print(f"     '{word}' → {idx}")
                continue

            elif user_input.lower() == 'model':
                # print(f"\nModel info:")
                # print(f"   Embedding: {predictor.model.embedding}")
                # print(f"   FC layer: {predictor.model.fc}")
                # print(f"   Classes: {predictor.class_names}")
                continue

            elif user_input.lower() == 'batch':
                # print("\n=== BATCH MODE ===")
                # print("Enter multiple texts (one per line). Type 'END' when done:")
                texts = []
                while True:
                    text = input("> ").strip()
                    if text.upper() == 'END':
                        break
                    if text:
                        texts.append(text)

                if texts:
                    results = predictor.predict_batch(texts)
                    # print("\n" + "=" * 60)
                    # print("BATCH RESULTS:")
                    # print("=" * 60)
                    # for i, result in enumerate(results):
                    #     print(f"\n{i+1}. 📄 Text: {result['text'][:50]}...")
                    #     if 'error' in result:
                    #         print(f"   Error: {result['error']}")
                    #     else:
                    #         print(f"   {result['predicted_class']} "
                    #               f"(Confidence: {result['confidence']:.1%})")
                continue

            elif user_input.lower() == '':
                continue

            # ===== SINGLE PREDICTION =====
            if show_details:
                result = predictor.predict(user_input, return_details=True)

                if 'error' in result:
                    # print(f"Error: {result['error']}")
                    pass
                else:
                    # print("\n" + "=" * 60)
                    # print(f"INPUT: {result['text']}")
                    # print(f"PREDICTION: {result['predicted_class']} "
                    #       f"(ID: {result['class_id']})")
                    # print(f"CONFIDENCE: {result['confidence']:.2%}")
                    # print(f"TOKENS: {result['num_tokens']} tokens")
                    # print("\nALL PROBABILITIES:")
                    # for class_name, prob in result['probabilities'].items():
                    #     bar_length = int(prob * 30)
                    #     bar = "█" * bar_length + "░" * (30 - bar_length)
                    #     print(f"  {class_name:15s} {prob:6.2%} {bar}")
                    # print("=" * 60)
                    pass
            else:
                predicted_class, confidence = predictor.predict(user_input)
                # print(f"→ {predicted_class} (Confidence: {confidence:.1%})")

        except KeyboardInterrupt:
            # print("Interrupted. Exiting...")
            break
        except Exception as e:
            # print(f"Error: {e}")
            # print("Please try again.")
            pass

def quick_test():
    """
    Quick test with example sentences
    """
    try:
        predictor = TextPredictor()
        
        test_sentences = [
            "I want to find a warm blue jacket",
            "I choose the first option",
            "I'll pay with my credit card",
            "Help me pick a red dress please",
            "Select the third item",
            "Use PayPal for payment"
        ]
        
        print("=" * 60)
        print("QUICK TEST")
        print("=" * 60)
        
        for sentence in test_sentences:
            predicted_class, confidence = predictor.predict(sentence)
            print(f"\n'{sentence}'")
            print(f"  → {predicted_class} ({confidence:.1%} confidence)")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")

ORDINAL_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "last": -1,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
}

PAYMENT_MAP = {
    "ovo": "OVO",
    "dana": "DANA",
    "gopay": "GOPAY",
    "shopeepay": "SHOPEEPAY",
    "bca": "BCA",
    "bri": "BRI",
    "mandiri": "MANDIRI",
    "bni": "BNI",
    "mastercard": "MASTERCARD",
    "visa": "VISA",
    "kredivo": "KREDIVO",
}

def extract_ordinal(text, threshold=60):
    tokens = text.lower().split()

    match, score, _ = process.extractOne(
        " ".join(tokens),
        ORDINAL_MAP.keys(),
        scorer=fuzz.partial_ratio
    )

    if score >= threshold:
        return ORDINAL_MAP[match]

    return None

def extract_payment(text, threshold=70):
    tokens = text.lower().split()

    match, score, _ = process.extractOne(
        " ".join(tokens),
        PAYMENT_MAP.keys(),
        scorer=fuzz.partial_ratio
    )

    if score >= threshold:
        return PAYMENT_MAP[match]

    return "UNSPECIFIED"


if __name__ == "__main__":
    import sys
    import json

    # CLI / SERVICE MODE
    if len(sys.argv) > 1:
        text = sys.argv[1]

        try:
            predictor = TextPredictor()
            predicted_class, confidence = predictor.predict(text)

            if predicted_class == "Select_Option":
                text = extract_ordinal(text)
            elif predicted_class == "Make_Payment":
                text = extract_payment(text)

            result = {
                "type": predicted_class.lower(),
                "confidence": float(confidence),
                "value": text
            }

            print(json.dumps(result))
            sys.exit(0)

        except Exception as e:
            print(json.dumps({
                "type": "unknown",
                "confidence": 0.0,
                "value": "",
                "error": str(e)
            }))
            sys.exit(1)

    # INTERACTIVE MODE (human)
    else:
        os.system('cls' if os.name == 'nt' else 'clear')
        interactive_test()