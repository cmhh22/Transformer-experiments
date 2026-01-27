"""
Utility functions for the BERT-Qwen Classification project
Includes model management, logging, reproducibility, and system utilities
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import hashlib
from contextlib import contextmanager
import time

# Configure module logger
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some operations don't have deterministic implementations
    else:
        torch.backends.cudnn.benchmark = True
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def setup_logging(log_dir: Optional[Path] = None,
                  log_level: int = logging.INFO,
                  log_to_file: bool = True,
                  log_to_console: bool = True,
                  experiment_name: str = "experiment") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        experiment_name: Name for the log file
    
    Returns:
        Configured root logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    return root_logger


def save_model(model: nn.Module, 
               tokenizer, 
               save_path: Path, 
               config,
               optimizer: Optional[torch.optim.Optimizer] = None,
               scheduler = None,
               epoch: Optional[int] = None,
               metrics: Optional[Dict[str, float]] = None):
    """
    Save model checkpoint with all training state
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        save_path: Path to save model
        config: Configuration object
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        metrics: Current metrics
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict() if hasattr(config, 'to_dict') else vars(config),
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # Save tokenizer
    tokenizer_path = save_path.parent / f"{save_path.stem}_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Tokenizer saved to: {tokenizer_path}")


def load_model(model_path: Path, 
               device: torch.device,
               model_class = None,
               load_optimizer: bool = False) -> Dict[str, Any]:
    """
    Load model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        model_class: Model class to instantiate (if None, returns state dict only)
        load_optimizer: Whether to return optimizer state
    
    Returns:
        Dictionary with model, config, and optionally optimizer state
    """
    model_path = Path(model_path)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    result = {
        'config': checkpoint.get('config', {}),
        'epoch': checkpoint.get('epoch'),
        'metrics': checkpoint.get('metrics'),
    }
    
    # Initialize model if class provided
    if model_class is not None:
        config = result['config']
        model_type = config.get('model_type', 'bert')
        num_classes = config.get('NUM_CLASSES', config.get('num_classes', 2))
        
        # Import here to avoid circular imports
        from model import get_model
        model = get_model(model_type=model_type, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        result['model'] = model
    else:
        result['model_state_dict'] = checkpoint['model_state_dict']
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    if 'scheduler_state_dict' in checkpoint:
        result['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
    
    logger.info(f"Model loaded from: {model_path}")
    
    return result


def count_parameters(model: nn.Module) -> Dict[str, int]:
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
        'non_trainable': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
    }


def print_model_summary(model: nn.Module, show_architecture: bool = False):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        show_architecture: Whether to print full architecture
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    param_counts = count_parameters(model)
    
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"Trainable ratio: {param_counts['trainable_ratio']:.2%}")
    
    if show_architecture:
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
    def convert_value(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            return v.item()
        elif isinstance(v, torch.Tensor):
            return v.cpu().item() if v.numel() == 1 else v.cpu().tolist()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        return v
    
    metrics_serializable = {k: convert_value(v) for k, v in metrics.items()}
    
    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    
    logger.info(f"Metrics saved to: {save_path}")


def load_metrics(load_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file"""
    with open(load_path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
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
    """Get information about available devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__,
    }
    
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        
        # Memory info
        info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        info['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    
    return info


def print_device_info():
    """Print device information"""
    info = get_device_info()
    
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"Device: {info['device_name']}")
        print(f"CUDA version: {info['cuda_version']}")
        print(f"cuDNN version: {info['cudnn_version']}")
        print(f"GPU Memory: {info['gpu_memory_total']:.2f} GB")
    
    print("="*60 + "\n")


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} completed in {format_time(elapsed)}")


def get_experiment_id(config: Any) -> str:
    """Generate unique experiment ID based on config"""
    config_str = json.dumps(config.to_dict() if hasattr(config, 'to_dict') else str(config), sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def freeze_layers(model: nn.Module, layers_to_freeze: List[str]):
    """
    Freeze specific layers of a model
    
    Args:
        model: PyTorch model
        layers_to_freeze: List of layer name patterns to freeze
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
            logger.debug(f"Froze layer: {name}")


def unfreeze_layers(model: nn.Module, layers_to_unfreeze: Optional[List[str]] = None):
    """
    Unfreeze specific or all layers of a model
    
    Args:
        model: PyTorch model
        layers_to_unfreeze: List of layer name patterns to unfreeze (None = all)
    """
    for name, param in model.named_parameters():
        if layers_to_unfreeze is None or any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True
            logger.debug(f"Unfroze layer: {name}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def cleanup_checkpoints(checkpoint_dir: Path, keep_n: int = 3, metric_name: str = 'val_loss'):
    """
    Clean up old checkpoints, keeping only the best N
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_n: Number of checkpoints to keep
        metric_name: Metric to use for selecting best checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if len(checkpoints) <= keep_n:
        return
    
    # Load metrics from each checkpoint
    checkpoint_metrics = []
    for ckpt in checkpoints:
        try:
            data = torch.load(ckpt, map_location='cpu')
            metrics = data.get('metrics', {})
            checkpoint_metrics.append((ckpt, metrics.get(metric_name, float('inf'))))
        except Exception:
            checkpoint_metrics.append((ckpt, float('inf')))
    
    # Sort by metric (lower is better for loss)
    checkpoint_metrics.sort(key=lambda x: x[1])
    
    # Remove checkpoints not in top N
    for ckpt, _ in checkpoint_metrics[keep_n:]:
        ckpt.unlink()
        # Also remove tokenizer folder if exists
        tokenizer_dir = checkpoint_dir / f"{ckpt.stem}_tokenizer"
        if tokenizer_dir.exists():
            import shutil
            shutil.rmtree(tokenizer_dir)
        logger.info(f"Removed checkpoint: {ckpt}")


def create_optimizer(model: nn.Module,
                     lr: float = 2e-5,
                     weight_decay: float = 0.01,
                     optimizer_type: str = 'adamw',
                     no_decay_params: Optional[List[str]] = None) -> torch.optim.Optimizer:
    """
    Create optimizer with proper weight decay handling
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer
        no_decay_params: Parameter names that should not have weight decay
    
    Returns:
        Configured optimizer
    """
    if no_decay_params is None:
        no_decay_params = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    # Separate parameters for weight decay
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay_params) and p.requires_grad],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay_params) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
