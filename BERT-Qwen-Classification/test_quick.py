"""
Quick tests to verify installation and basic functionality
Enhanced version with comprehensive testing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import numpy as np
        import pandas as pd
        import sklearn
        from tqdm import tqdm
        print("✓ All core packages imported successfully")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Transformers: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available, will use CPU")
        return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def test_transformers():
    """Test transformers library"""
    print("\nTesting transformers...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Try loading a small model
        print("  Loading DistilBERT tokenizer (smaller model)...")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Test tokenization
        text = "Hello, this is a test."
        tokens = tokenizer(text, return_tensors='pt')
        
        print(f"✓ Tokenization successful")
        print(f"  Input text: '{text}'")
        print(f"  Token shape: {tokens['input_ids'].shape}")
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        return False


def test_src_modules():
    """Test that src modules can be imported"""
    print("\nTesting src modules...")
    try:
        from config import Config, get_config
        from utils import set_seed, setup_logging
        from data_loader import TextDataset, load_data
        from model import get_model, BERTClassifier
        from train import train_epoch, validate
        from evaluate import evaluate_model, get_predictions
        
        config = get_config("bert")
        set_seed(42)
        
        print("✓ All src modules imported successfully")
        print(f"  Config: {config.model.model_name}")
        print(f"  Num classes: {config.model.num_classes}")
        return True
    except Exception as e:
        print(f"✗ src modules test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration system"""
    print("\nTesting configuration...")
    try:
        from config import Config, get_config, ModelConfig, TrainingConfig
        
        # Test dataclass config
        config = Config()
        
        # Test get_config helper
        bert_config = get_config("bert", num_classes=3, epochs=5)
        assert bert_config.model.num_classes == 3
        assert bert_config.training.epochs == 5
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        restored = Config.from_dict(config_dict)
        
        print("✓ Configuration system working")
        print(f"  Model: {config.model.model_name}")
        print(f"  Mixed precision: {config.training.use_mixed_precision}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test basic model creation"""
    print("\nTesting model creation...")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification
        
        print("  Creating DistilBERT model for classification...")
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False


def test_custom_model():
    """Test custom model architecture"""
    print("\nTesting custom model architecture...")
    try:
        from model import get_model, model_summary
        
        # Test model factory
        model = get_model(
            model_type='bert',
            model_name='distilbert-base-uncased',  # Use smaller model
            num_classes=3,
            dropout=0.2,
            pooling='mean'
        )
        
        summary = model_summary(model)
        print(f"✓ Custom model created successfully")
        print(f"  Pooling strategy: mean")
        
        return True
    except Exception as e:
        print(f"✗ Custom model test failed: {e}")
        return False


def test_forward_pass():
    """Test a forward pass through the model"""
    print("\nTesting forward pass...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Create model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        # Create dummy input
        text = "This is a test sentence for classification."
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        print(f"  Prediction: Class {pred} ({probs[0][pred]:.2%})")
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False


def test_data_loader():
    """Test data loading functionality"""
    print("\nTesting data loader...")
    try:
        from data_loader import load_data, TextDataset, TextAugmenter
        from transformers import AutoTokenizer
        
        # Load dummy data
        train_data, val_data, test_data = load_data(None)
        
        # Test TextDataset
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        dataset = TextDataset(
            train_data['text'].values[:10],
            train_data['label'].values[:10],
            tokenizer,
            max_length=128
        )
        
        sample = dataset[0]
        
        # Test augmenter
        augmenter = TextAugmenter(prob=0.3)
        original = "This is a test sentence."
        augmented = augmenter.augment(original)
        
        print(f"✓ Data loader working")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Sample input_ids shape: {sample['input_ids'].shape}")
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False


def test_mixed_precision():
    """Test mixed precision support"""
    print("\nTesting mixed precision...")
    try:
        import torch
        from torch.cuda.amp import autocast, GradScaler
        
        if torch.cuda.is_available():
            # Test autocast
            with autocast():
                x = torch.randn(10, 10, device='cuda')
                y = torch.matmul(x, x)
            
            scaler = GradScaler()
            print("✓ Mixed precision (AMP) supported")
        else:
            print("⚠ CUDA not available, skipping mixed precision test")
        
        return True
    except Exception as e:
        print(f"✗ Mixed precision test failed: {e}")
        return False


def test_evaluation():
    """Test evaluation utilities"""
    print("\nTesting evaluation utilities...")
    try:
        import numpy as np
        from evaluate import MetricsCalculator, calculate_metrics_per_class
        
        # Create dummy predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        y_prob = np.column_stack([1-y_pred, y_pred]).astype(float)
        
        # Test MetricsCalculator
        calculator = MetricsCalculator(num_classes=2, class_names=['Negative', 'Positive'])
        calculator.update(y_pred, y_true, y_prob)
        metrics = calculator.compute()
        
        print(f"✓ Evaluation utilities working")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("BERT-Qwen Classification - Quick Tests")
    print("="*60)
    
    tests = [
        ("Core Imports", test_imports),
        ("CUDA", test_cuda),
        ("Transformers", test_transformers),
        ("Source Modules", test_src_modules),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Custom Model", test_custom_model),
        ("Forward Pass", test_forward_pass),
        ("Data Loader", test_data_loader),
        ("Mixed Precision", test_mixed_precision),
        ("Evaluation", test_evaluation),
    ]
    
    results = []
    for name, test in tests:
        try:
            result = test()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
