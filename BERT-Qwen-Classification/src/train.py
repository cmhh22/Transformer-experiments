"""
Training functions with advanced features:
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling
- Early stopping with best model tracking
- Gradient clipping
- TensorBoard/WandB logging support
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
import time
import logging
from dataclasses import dataclass
from contextlib import nullcontext

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    loss: float
    accuracy: float
    learning_rate: float
    epoch: int
    step: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'epoch': self.epoch,
            'step': self.step
        }


class EarlyStopping:
    """Early stopping handler with best model tracking"""
    
    def __init__(self, 
                 patience: int = 3, 
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self._save_model(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    logger.info("Restored best model weights")
                return True
        else:
            self.best_score = score
            self._save_model(model)
            self.counter = 0
        
        return False
    
    def _save_model(self, model: nn.Module):
        """Save model state dict"""
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


class Trainer:
    """Advanced trainer class with all features"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 optimizer,
                 scheduler,
                 device: torch.device,
                 config,
                 loss_fn: Optional[nn.Module] = None,
                 callbacks: Optional[List[Callable]] = None):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Configuration object
            loss_fn: Loss function (default: CrossEntropyLoss)
            callbacks: Optional list of callback functions
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.callbacks = callbacks or []
        
        # Loss function with optional label smoothing
        label_smoothing = getattr(config.data, 'label_smoothing', 0.0) if hasattr(config, 'data') else 0.0
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Mixed precision
        self.use_amp = getattr(config.training, 'use_mixed_precision', True) if hasattr(config, 'training') else True
        self.scaler = GradScaler() if self.use_amp and device.type == 'cuda' else None
        
        # Gradient accumulation
        self.grad_accum_steps = getattr(config.training, 'gradient_accumulation_steps', 1) if hasattr(config, 'training') else 1
        
        # Early stopping
        patience = getattr(config.training, 'early_stopping_patience', 3) if hasattr(config, 'training') else getattr(config, 'EARLY_STOPPING_PATIENCE', 3)
        self.early_stopping = EarlyStopping(patience=patience)
        
        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with optional mixed precision
            amp_context = autocast() if self.use_amp and self.device.type == 'cuda' else nullcontext()
            
            with amp_context:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                loss = self.loss_fn(logits, labels)
                loss = loss / self.grad_accum_steps  # Scale for accumulation
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                max_grad_norm = getattr(self.config, 'MAX_GRAD_NORM', 1.0)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Metrics
            total_loss += loss.item() * self.grad_accum_steps
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%',
                'lr': f'{current_lr:.2e}'
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def fit(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            epochs: Number of epochs (uses config if not provided)
        
        Returns:
            Training history
        """
        epochs = epochs or getattr(self.config, 'EPOCHS', 10)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {self.grad_accum_steps}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Record history
            current_lr = self.scheduler.get_last_lr()[0]
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print summary
            logger.info(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Callbacks
            for callback in self.callbacks:
                callback(epoch, self.model, self.history)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/60:.1f} minutes")
        
        return self.history


# Backwards compatible functions
def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch (backwards compatible)"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # Mixed precision setup
    use_amp = getattr(config.training, 'use_mixed_precision', True) if hasattr(config, 'training') else getattr(config, 'USE_MIXED_PRECISION', True)
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision context
        amp_context = autocast() if use_amp and device.type == 'cuda' else nullcontext()
        
        with amp_context:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, 'MAX_GRAD_NORM', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, 'MAX_GRAD_NORM', 1.0))
            optimizer.step()
        
        scheduler.step()
        
        # Metrics
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        avg_loss = total_loss / (batch_idx + 1)
        avg_acc = correct_predictions.double() / total_samples
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{avg_acc:.4f}'
        })
    
    return total_loss / len(dataloader), correct_predictions.double() / total_samples


def validate(model, dataloader, device):
    """Validate model (backwards compatible)"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
    
    return total_loss / len(dataloader), correct_predictions.double() / total_samples


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config):
    """Full training loop (backwards compatible)"""
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    epochs = getattr(config, 'EPOCHS', 10)
    history = trainer.fit(epochs)
    
    return model, history
