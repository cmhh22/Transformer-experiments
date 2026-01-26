"""
Quick tests to verify installation and basic functionality
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
        print("✓ All core packages imported successfully")
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
        print("  Loading BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
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
        from config import Config
        from utils import set_seed
        
        config = Config()
        set_seed(42)
        
        print("✓ src modules imported successfully")
        print(f"  Config loaded: {config}")
        return True
    except Exception as e:
        print(f"✗ src modules test failed: {e}")
        return False


def test_model_creation():
    """Test basic model creation"""
    print("\nTesting model creation...")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification
        
        print("  Creating BERT model for classification...")
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
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


def test_forward_pass():
    """Test a forward pass through the model"""
    print("\nTesting forward pass...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Create model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )
        
        # Create dummy input
        text = "This is a test sentence."
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        print(f"  Probabilities: {probs[0].tolist()}")
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("BERT-Qwen Classification - Quick Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_cuda,
        test_transformers,
        test_src_modules,
        test_model_creation,
        test_forward_pass
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"⚠ {total - passed} test(s) failed")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
