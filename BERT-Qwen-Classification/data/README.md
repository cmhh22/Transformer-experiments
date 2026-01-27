# Data Directory

This directory is a placeholder for optional local datasets.

## Primary Dataset: HuggingFace Emotion

The main notebook uses the **Emotion** dataset directly from HuggingFace, which is automatically downloaded at runtime. **No local files needed!**

```python
from datasets import load_dataset
dataset = load_dataset("emotion")
```

### Emotion Dataset Details:
- **6 Classes**: sadness, joy, love, anger, fear, surprise
- **Train**: 16,000 samples
- **Validation**: 2,000 samples
- **Test**: 2,000 samples

---

## Optional: Custom Datasets

If you want to use custom datasets, place CSV files here:

### Expected Format:
```csv
text,label
"Sample text here",0
"Another sample",1
```

### Usage with CLI:
```bash
python main.py --data_path data/my_dataset.csv --text_column text --label_column label
```

## Supported Datasets

- ✅ **Emotion** (HuggingFace) - Primary, auto-downloaded
- IMDB Movie Reviews
- AG News
- Custom CSV/JSON files
