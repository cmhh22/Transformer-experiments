"""
Training functions and utilities
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Configuration object
    
    Returns:
        avg_loss: Average loss for the epoch
        avg_accuracy: Average accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_acc = correct_predictions.double() / total_samples
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{avg_acc:.4f}'
        })
    
    return total_loss / len(dataloader), correct_predictions.double() / total_samples


def validate(model, dataloader, device):
    """
    Validate model
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        device: Device to validate on
    
    Returns:
        avg_loss: Average validation loss
        avg_accuracy: Average validation accuracy
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = correct_predictions.double() / total_samples
    
    return avg_loss, avg_accuracy


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config):
    """
    Full training loop
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Configuration object
    
    Returns:
        model: Trained model
        history: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*60)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"✓ Validation loss improved! Saving model...")
            # Model will be saved at the end
        else:
            patience_counter += 1
            print(f"✗ No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    return model, history
