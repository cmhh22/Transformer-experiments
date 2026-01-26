"""
Configuration file for BERT-Qwen Classification
Contains all hyperparameters and settings
"""

import torch
from pathlib import Path


class Config:
    """Configuration class for model training and evaluation"""
    
    # Model settings
    MODEL_NAME = 'bert-base-uncased'  # or 'Qwen/Qwen-7B'
    NUM_CLASSES = 2  # Binary classification (adjust as needed)
    MAX_LENGTH = 512  # Maximum sequence length
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 0
    MAX_GRAD_NORM = 1.0
    
    # Data settings
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODEL_DIR = BASE_DIR / 'models'
    NOTEBOOK_DIR = BASE_DIR / 'notebooks'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42
    
    # Logging
    LOG_INTERVAL = 100  # Log every N batches
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 3
    
    # Evaluation
    EVAL_BATCH_SIZE = 64
    
    # Model-specific settings
    DROPOUT = 0.1
    HIDDEN_SIZE = 768  # BERT base hidden size
    NUM_ATTENTION_HEADS = 12
    NUM_HIDDEN_LAYERS = 12
    
    # Tokenizer settings
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    
    # Data augmentation (optional)
    USE_DATA_AUGMENTATION = False
    
    # Mixed precision training
    USE_MIXED_PRECISION = True
    
    def __repr__(self):
        """String representation of config"""
        config_str = "Config(\n"
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_str += f"  {key}={value},\n"
        config_str += ")"
        return config_str
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__class__.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
