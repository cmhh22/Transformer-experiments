"""
Model evaluation functions
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple


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
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, dim=1)
            
            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
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
    }
    
    # Add ROC-AUC for binary classification
    if all_probs.shape[1] == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            metrics['roc_auc'] = 0.0
    
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
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, dim=1)
            
            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probs)
    )


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names
    """
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
    print("="*60)


def calculate_metrics_per_class(y_true, y_pred, num_classes) -> Dict[int, Dict[str, float]]:
    """
    Calculate metrics for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        class_metrics: Dictionary with metrics per class
    """
    class_metrics = {}
    
    for class_idx in range(num_classes):
        # Binary mask for current class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        class_metrics[class_idx] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'support': np.sum(y_true_binary)
        }
    
    return class_metrics
