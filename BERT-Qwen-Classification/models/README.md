# Models Directory

This directory stores trained models and checkpoints.

## Contents

- `*.pth` - PyTorch model checkpoints
- `*_tokenizer/` - Saved tokenizer configurations
- `*.json` - Model metadata and configuration files

## Model Naming Convention

```
{model_type}_best_model.pth
{model_type}_epoch_{n}.pth
```

Where `model_type` can be:
- `bert` - BERT-based models
- `qwen` - Qwen-based models
- `attention` - Custom attention models

## Loading Models

```python
from src.utils import load_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, config = load_model('models/bert_best_model.pth', device)
```
