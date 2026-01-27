# BERT-Qwen Text Classification

## 📋 Description

Advanced text classification project using Transformer models (BERT, RoBERTa, DistilBERT, DeBERTa, ALBERT, and Qwen). Implements fine-tuning of pretrained models with production-ready features including mixed precision, gradient accumulation, early stopping, and comprehensive visualizations.

## 🎯 Objectives

- Fine-tuning multiple Transformer architectures for text classification
- Support for BERT, RoBERTa, DistilBERT, DeBERTa, ALBERT, and Qwen models
- Efficient training with mixed precision (FP16/BF16)
- Comparative performance analysis between models
- Comprehensive evaluation with advanced metrics and visualizations
- Production-ready code with logging, flexible configuration, and complete CLI

## 🚀 Features

### Supported Models
- **BERT** (base and large)
- **DistilBERT** (lightweight model)
- **RoBERTa** (robust training)
- **DeBERTa** (improved attention)
- **ALBERT** (shared parameters)
- **Qwen** (1.5B and 7B models)

### Training Features
- **Mixed Precision Training**: FP16/BF16 for faster training
- **Gradient Accumulation**: For larger effective batch sizes
- **Early Stopping**: With best model restoration
- **Multiple Schedulers**: Linear, Cosine, Constant
- **Data Augmentation**: Text augmentation techniques
- **Weighted Sampling**: For imbalanced datasets

### Evaluation & Visualization
- **Metrics**: Accuracy, F1, Precision, Recall, ROC-AUC, MCC, Cohen's Kappa
- **Visualizations**: Confusion Matrix, ROC Curve, PR Curve, Training History
- **Reports**: Automatic generation of comprehensive reports

## 📁 Project Structure

```
BERT-Qwen-Classification/
├── main.py                 # Main script with complete CLI
├── test_quick.py           # Quick validation tests
├── requirements.txt        # Updated dependencies
├── README.md              # Documentation
├── data/                  # 📂 Optional - datasets descargados localmente
├── models/                # 📂 Optional - modelos entrenados (.pth)
├── notebooks/             # Jupyter notebooks
│   └── BERT_vs_Qwen_Emotion_Analysis.ipynb  # ⭐ Main notebook
└── src/                   # Modular source code
    ├── __init__.py
    ├── config.py          # Configuration with dataclasses
    ├── model.py           # Model architectures
    ├── data_loader.py     # Data loading with augmentation
    ├── train.py           # Training with mixed precision
    ├── evaluate.py        # Evaluation with visualizations
    └── utils.py           # Advanced utilities
```

### 📌 Nota sobre Carpetas Opcionales

**`data/` - No necesaria:**
- El proyecto usa `load_dataset()` de HuggingFace
- Los datasets se descargan automáticamente desde la nube
- Se almacenan temporalmente en cache (~5 MB para Emotion dataset)

**`models/` - No necesaria:**
- Los archivos `.pth` se generan durante el entrenamiento
- **best_BERT.pt**: ~440 MB
- **best_Qwen.pt**: ~2 GB
- Solo descárgalos si vas a reutilizar los modelos entrenados
- Si ejecutas todo el notebook en una sesión, no necesitas descargarlos

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/cmhh22/transformer-experiments.git
cd transformer-experiments/BERT-Qwen-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Usage

### Basic Training

```bash
# Train BERT with default configuration
python main.py --model bert --epochs 10 --batch_size 32

# Train DistilBERT (faster)
python main.py --model distilbert --epochs 10 --batch_size 64

# Train RoBERTa
python main.py --model roberta --epochs 10 --batch_size 32
```

### Advanced Training

```bash
# With mixed precision and gradient accumulation
python main.py --model bert --epochs 10 --batch_size 16 --grad_accum_steps 4

# With cosine scheduler and warmup
python main.py --model bert --scheduler cosine --warmup_ratio 0.1

# With data augmentation
python main.py --model bert --use_augmentation

# Freeze base layers (transfer learning)
python main.py --model bert --freeze_base --learning_rate 1e-4

# With custom dataset
python main.py --model bert --data_path data/my_dataset.csv \
    --text_column review --label_column sentiment
```

### Evaluation

```bash
# Evaluate saved model
python main.py --mode eval --model_path models/bert_best.pth --data_path data/test.csv

# Generate complete report with visualizations
python main.py --mode eval --model_path models/bert_best.pth --verbose
```

### Interactive Prediction

```bash
# Interactive prediction mode
python main.py --mode predict --model_path models/bert_best.pth
```

### Quick Tests

```bash
# Verify installation
python test_quick.py
```

## 📈 Results

### Experimental Results (Emotion Classification)

Comparison of BERT vs Qwen on the **Emotion dataset** (6 classes) with 12,000 training samples:

| Model | Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **BERT-base** | 92.60% | 92.54% | 92.60% | **92.49%** | 5.9 min |
| **Qwen2.5-0.5B** | 92.60% | 92.77% | 92.60% | **92.63%** ⭐ | 34.3 min |

**🏆 Winner:** Qwen (F1 difference: 0.14%)

#### Per-Emotion Performance (F1 Scores)

| Emotion | BERT | Qwen |
|---------|------|------|
| Sadness | 96.65% | **97.29%** ⭐ |
| Joy | **93.98%** | 93.67% |
| Love | 79.70% | **79.74%** |
| Anger | 92.13% | **93.33%** ⭐ |
| Fear | **90.52%** | 89.47% |
| Surprise | 76.71% | 76.71% |

**Key Findings:**
- ⚡ **BERT**: 6x faster training, excellent for production
- 🎯 **Qwen**: Slightly better F1 (+0.14%), stronger on sadness & anger
- 📊 Both models: >92% accuracy, excellent generalization
- 💡 **Recommendation**: Use BERT for speed, Qwen for marginal accuracy gains

### Saved Artifacts

Results and artifacts are saved automatically during training:
- **`models/`**: Modelos entrenados (.pth) - ~440 MB (BERT) / ~2 GB (Qwen)
- **`logs/`**: Detailed training logs
- **`models/*_report/`**: Evaluation reports with visualizations
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `metrics.json`

**💡 Nota:** Si ejecutas el notebook completo en Colab en una sola sesión, los archivos `.pth` se generan, usan y pueden descartarse. Solo descárgalos si necesitas reutilizar los modelos entrenados.

## 🔧 Configuration

### Configuration via Code

```python
from src.config import Config, get_config

# Quick configuration by model type
config = get_config(
    model_type="bert",
    num_classes=3,
    batch_size=32,
    epochs=10,
    learning_rate=2e-5
)

# Access nested configuration
config.training.use_mixed_precision = True
config.training.gradient_accumulation_steps = 4
config.data.use_augmentation = True

# Save/load configuration
config.save("config.yaml")
config = Config.load("config.yaml")
```

### Configuration via CLI

```bash
python main.py --help  # See all available options
```

## 🧪 Code Architecture

### Main Modules

1. **config.py**: Configuration with dataclasses, validation, and YAML serialization
2. **model.py**: Flexible architectures with multiple pooling strategies
3. **data_loader.py**: Data loading with augmentation and caching
4. **train.py**: Advanced trainer with mixed precision and callbacks
5. **evaluate.py**: Complete metrics and visualization generation
6. **utils.py**: Utilities for logging, checkpointing, and optimization

### Production Features

- ✅ Complete type hints
- ✅ Structured logging
- ✅ Robust error handling
- ✅ Flexible configuration (CLI + code + YAML)
- ✅ Reproducibility (seeds, determinism)
- ✅ Checkpointing with full state
- ✅ Early stopping with restoration

## 📝 Usage Examples

### Sentiment Classification

```python
from transformers import AutoTokenizer
from src.config import get_config
from src.data_loader import load_data, create_dataloaders
from src.train import Trainer

# Configuration
config = get_config("bert", num_classes=3)  # Positive, Neutral, Negative

# Data
train_data, val_data, test_data = load_data("sentiment_data.csv")

# Training
trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, device, config)
history = trainer.fit(epochs=10)
```

## � Dataset

This project uses **online datasets** from HuggingFace:
- **Emotion** (HuggingFace) - 6 emotion classes ⭐ Primary
- IMDB Reviews
- AG News
- Custom datasets (CSV/JSON)

**Cómo funciona:**
- ☁️ `load_dataset("dair-ai/emotion")` descarga automáticamente desde HuggingFace
- 💾 Se cachea temporalmente en la VM (~5 MB)
- ⚡ No requiere carpeta `data/` local
- 🔄 Se vuelve a descargar en cada nueva sesión de Colab

## 🎓 Key Concepts

- **Transfer Learning**: Leveraging pretrained models
- **Attention Mechanism**: Attention mechanism in Transformers
- **Tokenization**: Text processing with specific tokenizers
- **Fine-tuning**: Adjusting pretrained models for specific tasks

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## 👤 Author

**Carlos Hernández**
- GitHub: [@cmhh22](https://github.com/cmhh22)
- LinkedIn: [Carlos Hernández](https://linkedin.com/in/cmhh22)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🚧 Project Status

🚀 Active development - January 2026
