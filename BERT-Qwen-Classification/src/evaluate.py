"""
Model evaluation functions with visualizations
Includes metrics calculation, visualization, and comprehensive reporting
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    logger.warning("matplotlib/seaborn not available. Visualization disabled.")


class MetricsCalculator:
    """Comprehensive metrics calculation for classification"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, preds: np.ndarray, labels: np.ndarray, probs: Optional[np.ndarray] = None):
        """Add batch predictions"""
        self.predictions.extend(preds.tolist())
        self.labels.extend(labels.tolist())
        if probs is not None:
            self.probabilities.extend(probs.tolist())
    
    def reset(self):
        """Reset stored predictions"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics"""
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities) if self.probabilities else None
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'support': len(y_true)
        }
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i, name in enumerate(self.class_names):
            mask_true = y_true == i
            mask_pred = y_pred == i
            metrics['per_class'][name] = {
                'precision': precision_score(mask_true, mask_pred, zero_division=0),
                'recall': recall_score(mask_true, mask_pred, zero_division=0),
                'f1': f1_score(mask_true, mask_pred, zero_division=0),
                'support': int(mask_true.sum())
            }
        
        # ROC-AUC for binary or multiclass
        if y_prob is not None and len(y_prob) > 0:
            try:
                if self.num_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
        
        return metrics


def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to evaluate on
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(all_labels, all_predictions),
    }
    
    # ROC-AUC for binary classification
    if all_probs.shape[1] == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_probs[:, 1])
            metrics['average_precision'] = average_precision_score(all_labels, all_probs[:, 1])
        except Exception:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
    
    return metrics


def get_predictions(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions from model
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to predict on
    
    Returns:
        predictions: Predicted labels
        labels: True labels
        probabilities: Class probabilities
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probs)
    )


def print_classification_report(y_true, y_pred, target_names=None):
    """Print detailed classification report"""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Additional metrics
    print(f"\nMatthews Correlation Coefficient: {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print("="*60)


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[Path] = None,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (10, 8)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for axis labels
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
        figsize: Figure size
    """
    if not HAS_VISUALIZATION:
        logger.warning("Visualization not available")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   save_path: Optional[Path] = None,
                   figsize: Tuple[int, int] = (10, 8)):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Class names for legend
        save_path: Path to save the plot
        figsize: Figure size
    """
    if not HAS_VISUALIZATION:
        logger.warning("Visualization not available")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        # Binary classification
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    else:
        # Multi-class
        n_classes = y_prob.shape[1]
        class_names = class_names or [f'Class {i}' for i in range(n_classes)]
        
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 save_path: Optional[Path] = None,
                                 figsize: Tuple[int, int] = (10, 8)):
    """Plot precision-recall curve for binary classification"""
    if not HAS_VISUALIZATION:
        logger.warning("Visualization not available")
        return
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, label=f'PR curve (AP = {ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"PR curve saved to {save_path}")
    
    plt.close()


def plot_training_history(history: Dict[str, List[float]],
                          save_path: Optional[Path] = None,
                          figsize: Tuple[int, int] = (14, 5)):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics over epochs
        save_path: Path to save the plot
        figsize: Figure size
    """
    if not HAS_VISUALIZATION:
        logger.warning("Visualization not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history.get('train_loss', []), 'b-', label='Train Loss', marker='o')
    axes[0].plot(epochs, history.get('val_loss', []), 'r-', label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history.get('train_acc', []), 'b-', label='Train Acc', marker='o')
    axes[1].plot(epochs, history.get('val_acc', []), 'r-', label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()


def generate_evaluation_report(model,
                                dataloader,
                                device,
                                class_names: Optional[List[str]] = None,
                                save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with visualizations
    
    Args:
        model: PyTorch model
        dataloader: Evaluation dataloader
        device: Device
        class_names: List of class names
        save_dir: Directory to save report and plots
    
    Returns:
        Complete evaluation report
    """
    # Get predictions
    predictions, labels, probabilities = get_predictions(model, dataloader, device)
    
    # Calculate metrics
    num_classes = probabilities.shape[1]
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(predictions, labels, probabilities)
    metrics = calculator.compute()
    
    # Add loss
    metrics['loss'] = evaluate_model(model, dataloader, device)['loss']
    
    # Save plots if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(labels, predictions, class_names, 
                             save_path=save_dir / 'confusion_matrix.png')
        
        # ROC curve
        plot_roc_curve(labels, probabilities, class_names,
                      save_path=save_dir / 'roc_curve.png')
        
        # PR curve (binary only)
        if num_classes == 2:
            plot_precision_recall_curve(labels, probabilities,
                                       save_path=save_dir / 'pr_curve.png')
        
        # Save metrics to JSON
        metrics_file = save_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {metrics_file}")
    
    # Print report
    print_classification_report(labels, predictions, class_names)
    
    return metrics


def calculate_metrics_per_class(y_true, y_pred, num_classes) -> Dict[int, Dict[str, float]]:
    """Calculate metrics for each class"""
    class_metrics = {}
    
    for class_idx in range(num_classes):
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        class_metrics[class_idx] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'support': int(np.sum(y_true_binary))
        }
    
    return class_metrics
