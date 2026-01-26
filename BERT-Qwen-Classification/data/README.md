# Data Directory

This directory contains datasets for BERT-Qwen classification.

## Structure

```
data/
├── raw/              # Original, unprocessed data
│   ├── train.csv     # Training data
│   ├── test.csv      # Test data
│   └── val.csv       # Validation data
└── processed/        # Processed and cleaned data
    ├── train.pkl     # Processed training data
    ├── test.pkl      # Processed test data
    └── val.pkl       # Processed validation data
```

## Data Format

Expected CSV format:
```csv
text,label
"Sample text here",0
"Another sample",1
```

## Supported Datasets

- IMDB Movie Reviews
- AG News
- Custom datasets (CSV/JSON format)

## Usage

Place your raw data files in the `raw/` directory. The preprocessing scripts will automatically create processed versions in the `processed/` directory.
