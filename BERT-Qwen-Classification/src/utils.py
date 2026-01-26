"""
Utility functions
"""

import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, tokenizer, save_path: Path, config):
    """
    Save model, tokenizer, and config
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        save_path: Path to save model
        config: Configuration object
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
    }, save_path)
    
    # Save tokenizer
    tokenizer_path = save_path.parent / f"{save_path.stem}_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"Model saved to: {save_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")


def load_model(model_path: Path, device):
    """
    Load model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Configuration
    """
    model_path = Path(model_path)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    
    # Initialize model architecture (need to know the architecture)
    from model import get_model
    
    # You may need to adjust this based on saved config
    model_type = config.get('model_type', 'bert')
    num_classes = config.get('NUM_CLASSES', 2)
    
    model = get_model(model_type=model_type, num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    
    return model, config


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        param_counts: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model):
    """
    Print model summary
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    param_counts = count_parameters(model)
    
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    
    print("\nModel architecture:")
    print(model)
    print("="*60 + "\n")


def save_metrics(metrics: Dict[str, Any], save_path: Path):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = value.item()
        else:
            metrics_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"Metrics saved to: {save_path}")


def load_metrics(load_path: Path) -> Dict[str, Any]:
    """
    Load metrics from JSON file
    
    Args:
        load_path: Path to metrics file
    
    Returns:
        metrics: Dictionary of metrics
    """
    load_path = Path(load_path)
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        formatted_time: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices
    
    Returns:
        device_info: Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info
