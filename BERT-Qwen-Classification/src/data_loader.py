"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    """Custom Dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of label integers
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_path: Optional[str] = None, 
              test_size: float = 0.2,
              val_size: float = 0.1,
              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train, validation, and test sets
    
    Args:
        data_path: Path to data file (CSV or JSON)
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random seed
    
    Returns:
        train_data, val_data, test_data
    """
    if data_path is None:
        # Create dummy data for testing
        print("No data path provided, creating dummy data...")
        texts = [
            "This is a positive example.",
            "This is a negative example.",
            "Another positive case here.",
            "Another negative case here.",
        ] * 100  # Repeat for more samples
        
        labels = [1, 0, 1, 0] * 100
        
        data = pd.DataFrame({
            'text': texts,
            'label': labels
        })
    else:
        # Load data from file
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            data = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Ensure required columns exist
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Data must contain 'text' and 'label' columns")
    
    # Split into train and test
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data['label']
    )
    
    # Split train into train and validation
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=train_val_data['label']
    )
    
    # Reset indices
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    return train_data, val_data, test_data


def create_dataloaders(train_data: Optional[pd.DataFrame],
                       val_data: Optional[pd.DataFrame],
                       test_data: Optional[pd.DataFrame],
                       tokenizer,
                       config) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        tokenizer: Tokenizer from transformers
        config: Configuration object
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = None
    val_loader = None
    test_loader = None
    
    if train_data is not None:
        train_dataset = TextDataset(
            train_data['text'].values,
            train_data['label'].values,
            tokenizer,
            config.MAX_LENGTH
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    
    if val_data is not None:
        val_dataset = TextDataset(
            val_data['text'].values,
            val_data['label'].values,
            tokenizer,
            config.MAX_LENGTH
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
    
    if test_data is not None:
        test_dataset = TextDataset(
            test_data['text'].values,
            test_data['label'].values,
            tokenizer,
            config.MAX_LENGTH
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
    
    return train_loader, val_loader, test_loader


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing
    
    Args:
        text: Input text string
    
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
