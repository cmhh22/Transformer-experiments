"""
Configuration file for BERT-Qwen Classification
Contains all hyperparameters and settings using dataclasses
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
import json
import yaml
from enum import Enum


class ModelType(str, Enum):
    """Supported model types"""
    BERT_BASE = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    DISTILBERT = "distilbert-base-uncased"
    ROBERTA = "roberta-base"
    ALBERT = "albert-base-v2"
    QWEN_1_5B = "Qwen/Qwen2.5-1.5B"
    QWEN_7B = "Qwen/Qwen2.5-7B"
    DEBERTA = "microsoft/deberta-v3-base"


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    model_name: str = ModelType.BERT_BASE.value
    num_classes: int = 2
    max_length: int = 512
    dropout: float = 0.1
    hidden_size: int = 768
    freeze_base: bool = False  # Freeze base model layers
    use_pooler: bool = True  # Use pooler output vs mean pooling
    classifier_hidden_size: Optional[int] = None  # Optional hidden layer in classifier
    

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # Proportion of training for warmup
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_mixed_precision: bool = True
    fp16: bool = True
    bf16: bool = False  # Use bf16 if available
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"
    
    # Scheduler
    scheduler_type: Literal["linear", "cosine", "constant", "polynomial"] = "linear"
    
    # Logging
    log_interval: int = 100
    eval_steps: Optional[int] = None  # Evaluate every N steps (None = per epoch)
    save_best_only: bool = True
    save_total_limit: int = 3  # Max checkpoints to keep


@dataclass
class DataConfig:
    """Data-specific configuration"""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Augmentation
    use_augmentation: bool = False
    augmentation_prob: float = 0.1
    
    # Processing
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    def __post_init__(self):
        """Validate splits"""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total}")


@dataclass 
class PathConfig:
    """Path configuration"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    def __post_init__(self):
        self.data_dir = self.base_dir / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.model_dir = self.base_dir / 'models'
        self.notebook_dir = self.base_dir / 'notebooks'
        self.log_dir = self.base_dir / 'logs'
        self.cache_dir = self.base_dir / '.cache'
        
    def create_dirs(self):
        """Create all necessary directories"""
        for dir_path in [self.data_dir, self.raw_data_dir, self.processed_data_dir,
                         self.model_dir, self.notebook_dir, self.log_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class combining all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # General settings
    seed: int = 42
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name: str = "bert_classification"
    
    # Backwards compatibility properties
    @property
    def EPOCHS(self):
        return self.training.epochs
    
    @EPOCHS.setter
    def EPOCHS(self, value):
        self.training.epochs = value
    
    @property
    def BATCH_SIZE(self):
        return self.training.batch_size
    
    @BATCH_SIZE.setter
    def BATCH_SIZE(self, value):
        self.training.batch_size = value
        
    @property
    def LEARNING_RATE(self):
        return self.training.learning_rate
    
    @LEARNING_RATE.setter  
    def LEARNING_RATE(self, value):
        self.training.learning_rate = value
        
    @property
    def MAX_LENGTH(self):
        return self.model.max_length
    
    @MAX_LENGTH.setter
    def MAX_LENGTH(self, value):
        self.model.max_length = value
        
    @property
    def NUM_CLASSES(self):
        return self.model.num_classes
    
    @NUM_CLASSES.setter
    def NUM_CLASSES(self, value):
        self.model.num_classes = value
        
    @property
    def MAX_GRAD_NORM(self):
        return self.training.max_grad_norm
    
    @property
    def DEVICE(self):
        return torch.device(self.device)
    
    @DEVICE.setter
    def DEVICE(self, value):
        self.device = str(value) if isinstance(value, torch.device) else value
    
    def __post_init__(self):
        """Initialize paths and validate config"""
        self.paths.create_dirs()
        self._validate()
    
    def _validate(self):
        """Validate configuration"""
        if self.training.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.model.dropout < 0 or self.model.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        result = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'seed': self.seed,
            'device': self.device,
            'experiment_name': self.experiment_name
        }
        # Convert Path objects to strings
        return json.loads(json.dumps(result, default=str))
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            experiment_name=config_dict.get('experiment_name', 'bert_classification')
        )
    
    def save(self, path: Path):
        """Save config to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def __repr__(self):
        """String representation"""
        return f"""Config(
  Model: {self.model.model_name}
  Classes: {self.model.num_classes}
  Epochs: {self.training.epochs}
  Batch Size: {self.training.batch_size}
  Learning Rate: {self.training.learning_rate}
  Device: {self.device}
  Experiment: {self.experiment_name}
)"""


# Convenience function for quick config creation
def get_config(model_type: str = "bert", **kwargs) -> Config:
    """
    Get config for a specific model type
    
    Args:
        model_type: "bert", "distilbert", "roberta", "qwen", etc.
        **kwargs: Override any config values
    
    Returns:
        Config object
    """
    model_mapping = {
        "bert": ModelType.BERT_BASE.value,
        "bert-large": ModelType.BERT_LARGE.value,
        "distilbert": ModelType.DISTILBERT.value,
        "roberta": ModelType.ROBERTA.value,
        "albert": ModelType.ALBERT.value,
        "qwen": ModelType.QWEN_1_5B.value,
        "qwen-7b": ModelType.QWEN_7B.value,
        "deberta": ModelType.DEBERTA.value,
    }
    
    model_name = model_mapping.get(model_type.lower(), model_type)
    
    config = Config()
    config.model.model_name = model_name
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
    
    return config
