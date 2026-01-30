# 🔬 BERT vs Qwen: Emotion Classification

## 📖 Overview

This project demonstrates **advanced fine-tuning techniques** for transformer models on emotion classification tasks. We compare two popular architectures — **BERT-base-uncased** (Google's bidirectional encoder) and **Qwen2.5-0.5B** (Alibaba's multilingual decoder) — on the **6-class Emotion dataset** from HuggingFace. 

The goal is to classify text into emotions: *sadness, joy, love, anger, fear, and surprise*. Through comprehensive experimentation, we evaluate model performance, training efficiency, and provide practical insights for choosing the right model for production scenarios. The project includes complete training pipelines, interpretable metrics, confusion matrices, and an educational Jupyter notebook ready for Google Colab.

---

## 🎯 Project Goals

Fine-tuning **BERT** and **Qwen** transformers for 6-class emotion classification using HuggingFace datasets.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📊 Experimental Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **BERT-base** | 92.60% | 92.54% | 92.60% | 92.49% | 5.9 min |
| **Qwen2.5-0.5B** | 92.60% | 92.77% | 92.60% | **92.63%** ⭐ | 34.3 min |

**🏆 Winner:** Qwen (by 0.14% F1) — but BERT is **6x faster**

### Per-Emotion F1 Scores

| Emotion | BERT | Qwen | Winner |
|---------|------|------|--------|
| Sadness | 96.65% | 97.29% | Qwen ⭐ |
| Joy | 93.98% | 93.67% | BERT |
| Love | 79.70% | 79.74% | Tie |
| Anger | 92.13% | 93.33% | Qwen ⭐ |
| Fear | 90.52% | 89.47% | BERT |
| Surprise | 76.71% | 76.71% | Tie |

### Key Insights
- ⚡ **Speed:** BERT trains 6x faster (ideal for production)
- 🎯 **Accuracy:** Nearly identical performance (0.14% difference)
- 💡 **Recommendation:** Use BERT for speed, Qwen for marginal gains on anger/sadness detection

---

## 🚀 Quick Start

### Google Colab (Recommended)
1. Upload `notebooks/BERT_vs_Qwen_Emotion_Analysis.ipynb` to Colab
2. Enable GPU: `Runtime > Change runtime type > GPU`
3. Run all cells: `Runtime > Run all`

**No local setup needed** — everything runs in the cloud!

### Local Setup (Optional)
```bash
git clone https://github.com/cmhh22/transformer-experiments.git
cd transformer-experiments/BERT-Qwen-Classification
pip install -r requirements.txt
python main.py --model bert --epochs 3
```

---

## 📁 Project Structure

```
BERT-Qwen-Classification/
├── notebooks/
│   └── BERT_vs_Qwen_Emotion_Analysis.ipynb  # ⭐ Main notebook (Colab-ready)
├── src/                     # Modular source code
│   ├── config.py            # Training configuration
│   ├── model.py             # TransformerClassifier architecture
│   ├── data_loader.py       # Dataset loading & preprocessing
│   ├── train.py             # Training loop with mixed precision
│   ├── evaluate.py          # Metrics & visualization
│   └── utils.py             # Helper utilities
├── main.py                  # CLI for local training
├── test_quick.py            # Quick validation tests
├── requirements.txt         # Dependencies
└── LICENSE                  # MIT License
```

---

## 🎯 Features

### Models
| Model | Parameters | Architecture | Pooling Strategy |
|-------|------------|--------------|------------------|
| BERT-base-uncased | ~110M | Encoder-only | CLS token |
| Qwen2.5-0.5B | ~500M | Decoder-only | Mean pooling |

### Dataset
- **Source:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (HuggingFace)
- **Classes:** 6 emotions (sadness, joy, love, anger, fear, surprise)
- **Samples:** 16,000 training / 2,000 test
- **Auto-download:** No manual download needed

### Training Features
- ✅ Mixed precision training (FP16)
- ✅ Cosine learning rate scheduling with warmup
- ✅ Gradient clipping for stability
- ✅ Early stopping with best model checkpointing
- ✅ Per-class and weighted metrics

### Visualizations
- 📊 Training curves (loss & F1)
- 📊 Confusion matrices
- 📊 Per-emotion performance comparison
- 📊 Side-by-side model comparison charts

---

## 🔧 Configuration

Key hyperparameters (modifiable in notebook):

```python
@dataclass
class Config:
    max_length: int = 128      # Max token length
    batch_size: int = 16       # BERT: 16, Qwen: 8
    epochs: int = 3            # Training epochs
    learning_rate: float = 2e-5  # BERT: 2e-5, Qwen: 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    use_amp: bool = True       # Mixed precision
```

---

## 📚 How It Works

### 1. Data Pipeline
```
HuggingFace Dataset → Tokenization → PyTorch DataLoader → Batches
```

### 2. Model Architecture
```
Input Text → Tokenizer → Transformer (BERT/Qwen) → Pooling → Classification Head → Emotion
```

### 3. Training Loop
```
Forward Pass → Loss (CrossEntropy) → Backward Pass → Optimizer Step → Scheduler Step
```

### 4. Evaluation
```
Predictions → Metrics (Accuracy, F1, Precision, Recall) → Confusion Matrix → Reports
```

---

## 📈 Training Details

### BERT Training
- **Batch size:** 16
- **Learning rate:** 2e-5
- **Pooling:** CLS token (first token)
- **Training time:** ~6 minutes on T4 GPU

### Qwen Training
- **Batch size:** 8 (larger model needs less)
- **Learning rate:** 1e-5 (more conservative for LLM)
- **Pooling:** Mean pooling (no CLS token in decoder models)
- **Training time:** ~34 minutes on T4 GPU

---

## 🛠️ CLI Usage (Local)

```bash
# Train BERT
python main.py --model bert --epochs 3 --batch_size 16

# Train with custom dataset
python main.py --model bert --data_path data/custom.csv --text_column text --label_column label

# Evaluate saved model
python main.py --mode eval --model_path models/bert_best.pth

# Interactive prediction
python main.py --mode predict --model_path models/bert_best.pth
```

---

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)

---

## 👤 Author

**Carlos Manuel Hernández**
- GitHub: [@cmhh22](https://github.com/cmhh22)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- HuggingFace for the Transformers library and Emotion dataset
- Google Colab for free GPU access
- The open-source ML community
