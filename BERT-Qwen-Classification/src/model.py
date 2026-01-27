"""
Model architectures and custom model implementations
Supports BERT, RoBERTa, DistilBERT, DeBERTa, ALBERT, and Qwen models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PoolingStrategy:
    """Different pooling strategies for sequence classification"""
    
    @staticmethod
    def cls_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Use [CLS] token representation"""
        return hidden_states[:, 0, :]
    
    @staticmethod
    def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean of all token representations (excluding padding)"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask
    
    @staticmethod
    def max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Max pooling over sequence (excluding padding)"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[mask_expanded == 0] = float('-inf')
        return torch.max(hidden_states, dim=1)[0]
    
    @staticmethod
    def attention_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor, 
                         attention_layer: nn.Module) -> torch.Tensor:
        """Learned attention-weighted pooling"""
        attention_scores = attention_layer(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        return torch.sum(hidden_states * attention_weights, dim=1)


class ClassificationHead(nn.Module):
    """Flexible classification head with optional hidden layer"""
    
    def __init__(self, 
                 input_size: int, 
                 num_classes: int, 
                 hidden_size: Optional[int] = None,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        if hidden_size is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            self.classifier = nn.Linear(input_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)


class BaseTransformerClassifier(nn.Module):
    """Base class for transformer-based classifiers"""
    
    def __init__(self, 
                 model_name: str,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 pooling: str = "cls",
                 freeze_base: bool = False,
                 classifier_hidden_size: Optional[int] = None,
                 gradient_checkpointing: bool = False):
        """
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Pooling strategy ("cls", "mean", "max", "attention")
            freeze_base: Whether to freeze base model weights
            classifier_hidden_size: Hidden size for classifier (None for linear)
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling
        
        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Freeze base model if requested
        if freeze_base:
            self._freeze_base_model()
        
        # Get hidden size
        hidden_size = self.config.hidden_size
        
        # Attention layer for attention pooling
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1)
            )
        else:
            self.attention = None
        
        # Classification head
        self.classifier = ClassificationHead(
            input_size=hidden_size,
            num_classes=num_classes,
            hidden_size=classifier_hidden_size,
            dropout=dropout
        )
        
        self._init_weights()
    
    def _freeze_base_model(self):
        """Freeze all parameters in the base transformer"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        logger.info(f"Base model frozen: {self.model_name}")
    
    def _unfreeze_base_model(self, num_layers: Optional[int] = None):
        """Unfreeze base model parameters"""
        for param in self.transformer.parameters():
            param.requires_grad = True
        
        if num_layers is not None:
            # Re-freeze all except last N layers
            self._freeze_base_model()
            if hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
                layers = self.transformer.encoder.layer
                for layer in layers[-num_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            logger.info(f"Unfroze last {num_layers} layers")
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy"""
        if self.pooling_strategy == "cls":
            return PoolingStrategy.cls_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == "mean":
            return PoolingStrategy.mean_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == "max":
            return PoolingStrategy.max_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == "attention":
            return PoolingStrategy.attention_pooling(hidden_states, attention_mask, self.attention)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional)
            return_hidden_states: Whether to return hidden states
        
        Returns:
            Dictionary with logits and optionally hidden states
        """
        # Prepare inputs
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None and 'token_type_ids' in self.transformer.forward.__code__.co_varnames:
            inputs['token_type_ids'] = token_type_ids
        
        # Get transformer outputs
        outputs = self.transformer(**inputs, output_hidden_states=return_hidden_states)
        
        # Handle different output formats
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        # Pool hidden states
        pooled_output = self._pool(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if return_hidden_states:
            result['hidden_states'] = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
            result['pooled_output'] = pooled_output
        
        return result
    
    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get attention weights from last layer"""
        with torch.no_grad():
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        return outputs.attentions[-1] if hasattr(outputs, 'attentions') else None


class BERTClassifier(BaseTransformerClassifier):
    """BERT-based classifier"""
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased', 
                 num_classes: int = 2, 
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            pooling=kwargs.get('pooling', 'cls'),
            freeze_base=kwargs.get('freeze_base', False),
            classifier_hidden_size=kwargs.get('classifier_hidden_size'),
            gradient_checkpointing=kwargs.get('gradient_checkpointing', False)
        )


class QwenClassifier(BaseTransformerClassifier):
    """Qwen-based classifier"""
    
    def __init__(self, 
                 model_name: str = 'Qwen/Qwen2.5-1.5B', 
                 num_classes: int = 2, 
                 dropout: float = 0.1,
                 **kwargs):
        # Qwen uses mean pooling by default since it doesn't have a pooler
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            pooling=kwargs.get('pooling', 'mean'),
            freeze_base=kwargs.get('freeze_base', False),
            classifier_hidden_size=kwargs.get('classifier_hidden_size'),
            gradient_checkpointing=kwargs.get('gradient_checkpointing', True)  # Enable by default for large models
        )


class MultiTaskClassifier(BaseTransformerClassifier):
    """Multi-task classifier with multiple output heads"""
    
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 task_configs: Dict[str, int] = None,
                 dropout: float = 0.1,
                 shared_hidden_size: int = 256,
                 **kwargs):
        """
        Args:
            model_name: HuggingFace model name
            task_configs: Dict mapping task names to num_classes
            dropout: Dropout rate
            shared_hidden_size: Size of shared hidden layer
        """
        # Initialize base with dummy num_classes
        super().__init__(
            model_name=model_name,
            num_classes=2,
            dropout=dropout,
            **kwargs
        )
        
        if task_configs is None:
            task_configs = {'main': 2}
        
        hidden_size = self.config.hidden_size
        
        # Replace single classifier with task-specific heads
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, shared_hidden_size),
            nn.GELU(),
            nn.LayerNorm(shared_hidden_size),
            nn.Dropout(dropout)
        )
        
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(shared_hidden_size, num_classes)
            for task, num_classes in task_configs.items()
        })
        
        # Remove the default classifier
        del self.classifier
        self.classifier = lambda x: x  # Placeholder
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                task: Optional[str] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task learning"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        pooled_output = self._pool(hidden_states, attention_mask)
        
        # Shared layer
        shared = self.shared_layer(pooled_output)
        
        # Get logits for specified task or all tasks
        if task is not None:
            return {'logits': self.task_heads[task](shared)}
        else:
            return {
                f'{task}_logits': head(shared) 
                for task, head in self.task_heads.items()
            }


def get_model(model_type: str = 'bert', 
              model_name: Optional[str] = None, 
              num_classes: int = 2, 
              dropout: float = 0.1,
              **kwargs) -> BaseTransformerClassifier:
    """
    Factory function to get model
    
    Args:
        model_type: Type of model ('bert', 'qwen', 'roberta', 'distilbert', 'deberta', 'albert')
        model_name: Specific model name (overrides default)
        num_classes: Number of classes
        dropout: Dropout rate
        **kwargs: Additional arguments passed to model
    
    Returns:
        model: PyTorch model
    """
    # Default model names for each type
    default_models = {
        'bert': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'roberta': 'roberta-base',
        'distilbert': 'distilbert-base-uncased',
        'deberta': 'microsoft/deberta-v3-base',
        'albert': 'albert-base-v2',
        'qwen': 'Qwen/Qwen2.5-1.5B',
        'qwen-7b': 'Qwen/Qwen2.5-7B',
    }
    
    # Get model name
    if model_name is None:
        model_name = default_models.get(model_type.lower())
        if model_name is None:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(default_models.keys())}")
    
    # Select appropriate class
    if 'qwen' in model_type.lower():
        return QwenClassifier(model_name, num_classes, dropout, **kwargs)
    else:
        return BERTClassifier(model_name, num_classes, dropout, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> str:
    """Generate model summary string"""
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    summary = [
        "=" * 60,
        "MODEL SUMMARY",
        "=" * 60,
        f"Model: {model.model_name if hasattr(model, 'model_name') else type(model).__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Non-trainable parameters: {total_params - trainable_params:,}",
        f"Trainable ratio: {100 * trainable_params / total_params:.2f}%",
        "=" * 60
    ]
    
    return "\n".join(summary)
