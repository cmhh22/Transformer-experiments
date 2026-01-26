"""
Model architectures and custom model implementations
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BERTClassifier(nn.Module):
    """Custom BERT-based classifier with additional layers"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.1):
        """
        Args:
            model_name: Name of pretrained model
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(BERTClassifier, self).__init__()
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Additional layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            logits: Class logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class QwenClassifier(nn.Module):
    """Custom Qwen-based classifier"""
    
    def __init__(self, model_name='Qwen/Qwen-7B', num_classes=2, dropout=0.1):
        """
        Args:
            model_name: Name of pretrained Qwen model
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(QwenClassifier, self).__init__()
        
        # Load pretrained Qwen
        self.qwen = AutoModel.from_pretrained(model_name)
        self.config = self.qwen.config
        
        # Get hidden size
        hidden_size = self.config.hidden_size
        
        # Additional layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            logits: Class logits
        """
        # Get Qwen outputs
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use last hidden state (take mean or first token)
        # For Qwen, we'll use the last token's representation
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class AttentionClassifier(nn.Module):
    """
    Classifier with attention mechanism on top of transformer
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.1):
        """
        Args:
            model_name: Name of pretrained model
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(AttentionClassifier, self).__init__()
        
        # Load pretrained model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass with attention
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            logits: Class logits
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Compute attention weights
        attention_scores = self.attention(last_hidden_state)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float('-inf')
        )
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum of hidden states
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        weighted_output = torch.sum(
            last_hidden_state * attention_weights, dim=1
        )  # (batch, hidden)
        
        # Apply dropout
        weighted_output = self.dropout(weighted_output)
        
        # Classification
        logits = self.classifier(weighted_output)
        
        return logits


def get_model(model_type='bert', model_name=None, num_classes=2, dropout=0.1):
    """
    Factory function to get model
    
    Args:
        model_type: Type of model ('bert', 'qwen', 'attention')
        model_name: Specific model name (overrides default)
        num_classes: Number of classes
        dropout: Dropout rate
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'bert':
        if model_name is None:
            model_name = 'bert-base-uncased'
        return BERTClassifier(model_name, num_classes, dropout)
    
    elif model_type == 'qwen':
        if model_name is None:
            model_name = 'Qwen/Qwen-7B'
        return QwenClassifier(model_name, num_classes, dropout)
    
    elif model_type == 'attention':
        if model_name is None:
            model_name = 'bert-base-uncased'
        return AttentionClassifier(model_name, num_classes, dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
