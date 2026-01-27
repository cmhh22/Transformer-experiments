"""
Data loading and preprocessing utilities
Includes data augmentation, caching, and efficient loading
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import re
import logging
import hashlib
import pickle

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Text augmentation techniques for NLP"""
    
    def __init__(self, prob: float = 0.1):
        self.prob = prob
        
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap words in text"""
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n):
            if random.random() < self.prob:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        if len(words) == 1:
            return text
        remaining = [w for w in words if random.random() > p]
        if len(remaining) == 0:
            return random.choice(words)
        return ' '.join(remaining)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert copies of words"""
        words = text.split()
        for _ in range(n):
            if random.random() < self.prob and len(words) > 0:
                word = random.choice(words)
                idx = random.randint(0, len(words))
                words.insert(idx, word)
        return ' '.join(words)
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Placeholder for synonym replacement (requires wordnet)"""
        # In production, use nltk.wordnet for actual synonyms
        return text
    
    def back_translation_simulation(self, text: str) -> str:
        """Simulate back-translation by word shuffling"""
        words = text.split()
        if len(words) <= 3:
            return text
        # Shuffle middle words
        middle = words[1:-1]
        random.shuffle(middle)
        return ' '.join([words[0]] + middle + [words[-1]])
    
    def augment(self, text: str, methods: Optional[List[str]] = None) -> str:
        """Apply random augmentation"""
        if methods is None:
            methods = ['swap', 'deletion', 'insertion']
        
        method = random.choice(methods)
        
        if method == 'swap':
            return self.random_swap(text)
        elif method == 'deletion':
            return self.random_deletion(text, p=self.prob)
        elif method == 'insertion':
            return self.random_insertion(text)
        elif method == 'back_translation':
            return self.back_translation_simulation(text)
        else:
            return text


class TextDataset(Dataset):
    """Enhanced Dataset for text classification with augmentation support"""
    
    def __init__(self, 
                 texts: Union[List[str], np.ndarray],
                 labels: Union[List[int], np.ndarray],
                 tokenizer,
                 max_length: int = 512,
                 augment: bool = False,
                 augment_prob: float = 0.1,
                 cache_tokenization: bool = False,
                 preprocessing_fn: Optional[Callable[[str], str]] = None):
        """
        Args:
            texts: List of text strings
            labels: List of label integers
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length
            augment: Whether to apply augmentation
            augment_prob: Probability of augmentation
            cache_tokenization: Cache tokenized inputs
            preprocessing_fn: Optional preprocessing function
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = TextAugmenter(prob=augment_prob) if augment else None
        self.preprocessing_fn = preprocessing_fn
        self.cache_tokenization = cache_tokenization
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def _preprocess(self, text: str) -> str:
        """Apply preprocessing"""
        if self.preprocessing_fn:
            text = self.preprocessing_fn(text)
        return text
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text"""
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
            'attention_mask': encoding['attention_mask'].flatten()
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Preprocessing
        text = self._preprocess(text)
        
        # Augmentation (only during training)
        if self.augment and self.augmenter:
            text = self.augmenter.augment(text)
            # Don't cache augmented text
            encoding = self._tokenize(text)
        else:
            # Check cache
            if self.cache_tokenization and idx in self._cache:
                encoding = self._cache[idx]
            else:
                encoding = self._tokenize(text)
                if self.cache_tokenization:
                    self._cache[idx] = encoding
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get label distribution"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class DynamicBatchDataset(TextDataset):
    """Dataset with dynamic batching by sequence length for efficiency"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute approximate lengths for sorting
        self.lengths = [len(str(t).split()) for t in self.texts]
    
    def get_sorted_indices(self) -> List[int]:
        """Get indices sorted by length"""
        return sorted(range(len(self)), key=lambda i: self.lengths[i])


def load_data(data_path: Optional[str] = None, 
              test_size: float = 0.2,
              val_size: float = 0.1,
              random_state: int = 42,
              text_column: str = 'text',
              label_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train, validation, and test sets
    
    Args:
        data_path: Path to data file (CSV, JSON, or Parquet)
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random seed
        text_column: Name of text column
        label_column: Name of label column
    
    Returns:
        train_data, val_data, test_data
    """
    if data_path is None:
        # Create dummy data for testing
        logger.info("No data path provided, creating dummy data...")
        texts = [
            "This product is absolutely amazing! I love it.",
            "Terrible experience. Would not recommend to anyone.",
            "Great quality and fast shipping. Very satisfied!",
            "Waste of money. Broke after one day of use.",
            "Exceeded my expectations. Will buy again.",
            "Poor customer service. Very disappointed.",
            "Best purchase I've made this year!",
            "Complete garbage. Don't waste your time.",
        ] * 50  # Repeat for more samples
        
        labels = [1, 0, 1, 0, 1, 0, 1, 0] * 50
        
        data = pd.DataFrame({
            text_column: texts,
            label_column: labels
        })
    else:
        # Load data from file
        data_path = Path(data_path)
        logger.info(f"Loading data from {data_path}")
        
        if data_path.suffix == '.csv':
            data = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            data = pd.read_json(data_path)
        elif data_path.suffix == '.parquet':
            data = pd.read_parquet(data_path)
        elif data_path.suffix in ['.tsv', '.txt']:
            data = pd.read_csv(data_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Ensure required columns exist
    if text_column not in data.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {list(data.columns)}")
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(data.columns)}")
    
    # Rename columns to standard names
    data = data.rename(columns={text_column: 'text', label_column: 'label'})
    
    # Clean data
    data = data.dropna(subset=['text', 'label'])
    data['text'] = data['text'].astype(str)
    
    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Label distribution: {data['label'].value_counts().to_dict()}")
    
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
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def create_dataloaders(train_data: Optional[pd.DataFrame],
                       val_data: Optional[pd.DataFrame],
                       test_data: Optional[pd.DataFrame],
                       tokenizer,
                       config,
                       use_weighted_sampler: bool = False) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        tokenizer: Tokenizer from transformers
        config: Configuration object
        use_weighted_sampler: Use weighted sampling for imbalanced data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = None
    val_loader = None
    test_loader = None
    
    # Get config values with backwards compatibility
    batch_size = getattr(config, 'BATCH_SIZE', None) or getattr(config.training, 'batch_size', 32)
    eval_batch_size = getattr(config, 'EVAL_BATCH_SIZE', None) or getattr(config.training, 'eval_batch_size', 64)
    max_length = getattr(config, 'MAX_LENGTH', None) or getattr(config.model, 'max_length', 512)
    num_workers = getattr(config.data, 'num_workers', 0) if hasattr(config, 'data') else 0
    pin_memory = getattr(config.data, 'pin_memory', True) if hasattr(config, 'data') else True
    use_augmentation = getattr(config.data, 'use_augmentation', False) if hasattr(config, 'data') else False
    
    # Windows compatibility
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
    
    if train_data is not None:
        train_dataset = TextDataset(
            train_data['text'].values,
            train_data['label'].values,
            tokenizer,
            max_length,
            augment=use_augmentation
        )
        
        sampler = None
        shuffle = True
        
        if use_weighted_sampler:
            # Create weighted sampler for imbalanced data
            label_counts = train_dataset.get_label_distribution()
            weights = [1.0 / label_counts[l] for l in train_data['label'].values]
            sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop last incomplete batch for stable training
        )
    
    if val_data is not None:
        val_dataset = TextDataset(
            val_data['text'].values,
            val_data['label'].values,
            tokenizer,
            max_length,
            augment=False  # No augmentation for validation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    if test_data is not None:
        test_dataset = TextDataset(
            test_data['text'].values,
            test_data['label'].values,
            tokenizer,
            max_length,
            augment=False  # No augmentation for testing
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader


def get_kfold_splits(data: pd.DataFrame, 
                     n_splits: int = 5, 
                     random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Get K-Fold cross-validation splits
    
    Args:
        data: DataFrame with 'text' and 'label' columns
        n_splits: Number of folds
        random_state: Random seed
    
    Returns:
        List of (train_df, val_df) tuples
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, val_idx in kfold.split(data, data['label']):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)
        splits.append((train_df, val_df))
    
    return splits


def preprocess_text(text: str, 
                   lowercase: bool = True,
                   remove_urls: bool = True,
                   remove_special_chars: bool = False,
                   remove_numbers: bool = False) -> str:
    """
    Text preprocessing pipeline
    
    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_urls: Remove URLs
        remove_special_chars: Remove special characters
        remove_numbers: Remove numbers
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def cache_dataset(dataset: TextDataset, cache_path: Path) -> None:
    """Cache a tokenized dataset to disk"""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Force tokenization of all samples
    cached_data = []
    for i in range(len(dataset)):
        cached_data.append(dataset[i])
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_data, f)
    
    logger.info(f"Cached {len(cached_data)} samples to {cache_path}")


def load_cached_dataset(cache_path: Path) -> Optional[List[Dict[str, torch.Tensor]]]:
    """Load cached dataset from disk"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} samples from cache")
        return data
    return None
