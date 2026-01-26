"""
Main script for BERT-Qwen Classification
Handles training, evaluation and inference of transformer models
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from data_loader import load_data, create_dataloaders
from train import train_model
from evaluate import evaluate_model
from utils import set_seed, save_model, load_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BERT-Qwen Classification")
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'predict'],
                       help='Mode: train, eval or predict')
    
    # Model
    parser.add_argument('--model', type=str, default='bert',
                       choices=['bert', 'qwen'],
                       help='Model type: bert or qwen')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for evaluation/prediction')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    
    # Data
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset')
    
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Update config with args
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.MAX_LENGTH = args.max_length
    config.DEVICE = device
    
    # Select model name based on model type
    if args.model == 'bert':
        model_name = 'bert-base-uncased'
    elif args.model == 'qwen':
        model_name = 'Qwen/Qwen-7B'
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    print(f"\n{'='*60}")
    print(f"BERT-Qwen Classification - {args.mode.upper()} Mode")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if args.mode == 'train':
        # Load data
        print("Loading training data...")
        train_data, val_data, test_data = load_data(args.data_path)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data, val_data, test_data, 
            tokenizer, config
        )
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Initialize model
        print(f"\nInitializing {model_name} model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.NUM_CLASSES
        )
        model.to(device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        total_steps = len(train_loader) * config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Train model
        print("\nStarting training...")
        trained_model, history = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, device, config
        )
        
        # Save model
        save_path = Path(args.save_dir) / f"{args.model}_best_model.pth"
        save_model(trained_model, tokenizer, save_path, config)
        print(f"\nModel saved to: {save_path}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(trained_model, test_loader, device)
        
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("--model_path required for evaluation mode")
        
        # Load model
        print(f"Loading model from {args.model_path}...")
        model, _ = load_model(args.model_path, device)
        
        # Load test data
        print("Loading test data...")
        _, _, test_data = load_data(args.data_path)
        _, _, test_loader = create_dataloaders(
            None, None, test_data,
            tokenizer, config
        )
        
        # Evaluate
        print("Evaluating model...")
        metrics = evaluate_model(model, test_loader, device)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    elif args.mode == 'predict':
        if args.model_path is None:
            raise ValueError("--model_path required for prediction mode")
        
        # Load model
        print(f"Loading model from {args.model_path}...")
        model, _ = load_model(args.model_path, device)
        
        # Interactive prediction loop
        print("\n" + "="*60)
        print("PREDICTION MODE - Enter text to classify (or 'quit' to exit)")
        print("="*60)
        
        model.eval()
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item()
            
            print(f"Prediction: Class {pred_class}")
            print(f"Confidence: {confidence:.4f}")
            print(f"All probabilities: {probs[0].cpu().numpy()}")
    
    print("\n" + "="*60)
    print("Process completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
