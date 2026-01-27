# 📓 Notebooks Directory

**Self-contained notebook for Google Colab** - No external imports needed!

## Main Notebook

| Notebook | Description | Models | Colab Ready |
|----------|-------------|--------|-------------|
| **[BERT_vs_Qwen_Emotion_Analysis.ipynb](BERT_vs_Qwen_Emotion_Analysis.ipynb)** | ⭐ Emotion classification with HuggingFace dataset | BERT + Qwen | ✅ Yes |

---

## 🔬 BERT_vs_Qwen_Emotion_Analysis.ipynb

**Complete notebook** for emotion sentiment analysis using the official **Emotion** dataset from HuggingFace (6 emotion classes).

### Key Features:
- ✅ **Real Dataset**: Emotion from HuggingFace (6 classes: sadness, joy, love, anger, fear, surprise)
- ✅ **Multiclass Classification**: 6 emotion categories
- ✅ **Complete Fine-tuning**: BERT-base-uncased and Qwen2.5-0.5B
- ✅ **Side-by-side Comparison**: Performance metrics and visualizations
- ✅ **Confusion Matrices**: Heatmaps for both models
- ✅ **Full Metrics**: Accuracy, Precision, Recall, F1 (weighted and per-class)
- ✅ **Inference**: Test emotion prediction on custom texts
- ✅ **Fully Commented**: All code with English comments

### Expected Results:
```
+-------+----------+-----------+--------+----------+
| Model | Accuracy | Precision | Recall | F1 Score |
+-------+----------+-----------+--------+----------+
| BERT  |  ~0.92   |   ~0.92   | ~0.92  |   ~0.92  |
| Qwen  |  ~0.90   |   ~0.90   | ~0.90  |   ~0.90  |
+-------+----------+-----------+--------+----------+
```

---

## 🚀 How to Use

### Option 1: Google Colab (Recommended)
1. Upload the notebook to Google Colab
2. Select GPU runtime: `Runtime > Change runtime type > GPU`
3. Run all cells: `Runtime > Run all`

### Option 2: Local Jupyter
```bash
pip install -r ../requirements.txt
jupyter notebook
```

## 📊 What's Included

- ✅ Dependency installation
- ✅ Dataset loading from HuggingFace
- ✅ Model architecture definition
- ✅ Training loop with visualizations
- ✅ Evaluation metrics & confusion matrix
- ✅ Inference examples
- ✅ Model comparison

## Requirements (auto-installed in Colab)

```
transformers>=4.40.0
datasets
torch>=2.0.0
accelerate
scikit-learn
matplotlib
seaborn
tqdm
```
