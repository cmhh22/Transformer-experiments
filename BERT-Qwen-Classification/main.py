"""
Main script for BERT-Qwen Classification
Advanced transformer-based text classification with support for multiple models,
mixed precision training, gradient accumulation, and comprehensive logging.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config, get_config, ModelType
from data_loader import load_data, create_dataloaders
from train import train_model, Trainer
from evaluate import evaluate_model, generate_evaluation_report, plot_training_history
from utils import (
    set_seed, save_model, load_model, setup_logging, 
    print_model_summary, print_device_info, timer, create_optimizer
)

# Setup logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BERT-Qwen Classification - Advanced Transformer Text Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'predict', 'export'],
                        help='Mode: train, eval, predict or export')
    
    # Model selection
    parser.add_argument('--model', type=str, default='bert',
                        choices=['bert', 'bert-large', 'distilbert', 'roberta', 
                                 'deberta', 'albert', 'qwen', 'qwen-7b'],
                        help='Model type to use')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for evaluation/prediction')
    
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for regularization')
    
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Ratio of training steps for warmup')
    
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'cosine', 'constant'],
                        help='Learning rate scheduler')
    
    # Data
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (CSV, JSON, or Parquet)')
    
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of the text column in dataset')
    
    parser.add_argument('--label_column', type=str, default='label',
                        help='Name of the label column in dataset')
    
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation during training')
    
    # Advanced options
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base model weights')
    
    parser.add_argument('--pooling', type=str, default='cls',
                        choices=['cls', 'mean', 'max', 'attention'],
                        help='Pooling strategy for classification')
    
    parser.add_argument('--early_stopping', type=int, default=3,
                        help='Early stopping patience (0 to disable)')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def get_scheduler(optimizer, scheduler_type: str, num_training_steps: int, warmup_ratio: float):
    """Get learning rate scheduler"""
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:  # constant
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )


def main():
    """Main execution function"""
    args = parse_args()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.model}_{args.mode}_{timestamp}"
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(
        log_dir=Path(args.log_dir),
        log_level=log_level,
        experiment_name=args.experiment_name
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    print_device_info()
    
    # Create config
    config = get_config(
        model_type=args.model,
        num_classes=args.num_classes,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Override config with args
    config.training.batch_size = args.batch_size
    config.training.eval_batch_size = args.eval_batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.warmup_ratio = args.warmup_ratio
    config.training.gradient_accumulation_steps = args.grad_accum_steps
    config.training.use_mixed_precision = args.use_amp and not args.no_amp
    config.training.early_stopping_patience = args.early_stopping
    config.training.scheduler_type = args.scheduler
    config.model.max_length = args.max_length
    config.model.freeze_base = args.freeze_base
    config.data.use_augmentation = args.use_augmentation
    config.device = str(device)
    config.experiment_name = args.experiment_name
    
    # Get model name
    model_name = config.model.model_name
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BERT-Qwen Classification - {args.mode.upper()} Mode")
    logger.info(f"Model: {model_name}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"{'='*60}\n")
    logger.info(f"Configuration:\n{config}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Handle tokenizers without padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.mode == 'train':
        with timer("Training"):
            # Load data
            logger.info("Loading training data...")
            train_data, val_data, test_data = load_data(
                args.data_path,
                text_column=args.text_column,
                label_column=args.label_column
            )
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_data, val_data, test_data,
                tokenizer, config
            )
            
            logger.info(f"Training samples: {len(train_data)}")
            logger.info(f"Validation samples: {len(val_data)}")
            logger.info(f"Test samples: {len(test_data)}")
            
            # Initialize model
            logger.info(f"\nInitializing {model_name} model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=config.model.num_classes,
                trust_remote_code=True
            )
            model.to(device)
            
            # Print model summary
            print_model_summary(model, show_architecture=args.verbose)
            
            # Freeze base if requested
            if args.freeze_base:
                for name, param in model.named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False
                logger.info("Base model layers frozen")
            
            # Setup optimizer with proper weight decay
            optimizer = create_optimizer(
                model,
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
            
            # Setup scheduler
            total_steps = len(train_loader) * config.training.epochs // config.training.gradient_accumulation_steps
            scheduler = get_scheduler(
                optimizer,
                config.training.scheduler_type,
                total_steps,
                config.training.warmup_ratio
            )
            
            # Train model
            logger.info("\nStarting training...")
            trained_model, history = train_model(
                model, train_loader, val_loader,
                optimizer, scheduler, device, config
            )
            
            # Save model
            save_dir = Path(args.save_dir)
            save_path = save_dir / f"{args.experiment_name}_best.pth"
            save_model(trained_model, tokenizer, save_path, config)
            
            # Save training history plot
            plot_training_history(history, save_path=save_dir / f"{args.experiment_name}_history.png")
            
            # Evaluate on test set
            logger.info("\nEvaluating on test set...")
            test_metrics = evaluate_model(trained_model, test_loader, device)
            
            logger.info("\n" + "="*60)
            logger.info("FINAL TEST RESULTS")
            logger.info("="*60)
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Generate full evaluation report
            report_dir = save_dir / f"{args.experiment_name}_report"
            generate_evaluation_report(
                trained_model, test_loader, device,
                save_dir=report_dir
            )
    
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("--model_path required for evaluation mode")
        
        with timer("Evaluation"):
            # Load model
            logger.info(f"Loading model from {args.model_path}...")
            result = load_model(args.model_path, device)
            model = result.get('model')
            
            if model is None:
                # Need to create model and load state dict
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=config.model.num_classes,
                    trust_remote_code=True
                )
                model.load_state_dict(result['model_state_dict'])
                model.to(device)
                model.eval()
            
            # Load test data
            logger.info("Loading test data...")
            _, _, test_data = load_data(
                args.data_path,
                text_column=args.text_column,
                label_column=args.label_column
            )
            _, _, test_loader = create_dataloaders(
                None, None, test_data,
                tokenizer, config
            )
            
            # Generate evaluation report
            report_dir = Path(args.save_dir) / f"{args.experiment_name}_eval"
            metrics = generate_evaluation_report(
                model, test_loader, device,
                save_dir=report_dir
            )
            
            logger.info("\n" + "="*60)
            logger.info("EVALUATION RESULTS")
            logger.info("="*60)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.4f}")
    
    elif args.mode == 'predict':
        if args.model_path is None:
            raise ValueError("--model_path required for prediction mode")
        
        # Load model
        logger.info(f"Loading model from {args.model_path}...")
        result = load_model(args.model_path, device)
        model = result.get('model')
        
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=config.model.num_classes,
                trust_remote_code=True
            )
            model.load_state_dict(result['model_state_dict'])
            model.to(device)
            model.eval()
        
        # Interactive prediction loop
        print("\n" + "="*60)
        print("PREDICTION MODE - Enter text to classify (or 'quit' to exit)")
        print("="*60)
        
        while True:
            try:
                text = input("\nEnter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=config.model.max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item()
            
            print(f"\n📊 Prediction: Class {pred_class}")
            print(f"📈 Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
            print(f"📉 All probabilities: {probs[0].cpu().numpy()}")
    
    elif args.mode == 'export':
        if args.model_path is None:
            raise ValueError("--model_path required for export mode")
        
        logger.info(f"Exporting model from {args.model_path}...")
        # Export to ONNX or other formats could be added here
        logger.info("Export functionality coming soon!")
    
    logger.info("\n" + "="*60)
    logger.info("Process completed successfully!")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
        
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
