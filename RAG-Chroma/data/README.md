# RAG-Chroma Data Directory

This directory contains source documents for the RAG system.

## Structure
```
data/
├── README.md           # This file
├── sample_docs/        # Sample documents
├── raw/               # Original documents
└── processed/         # Processed documents (optional)
```

## Supported Formats
- **PDF** (.pdf): PDF documents
- **Text** (.txt): Plain text files
- **Markdown** (.md): Markdown files
- **Word** (.docx): Microsoft Word documents

## Usage
1. Place your documents in this directory or subdirectories
2. The system will automatically process them when running `main.py`
3. Documents will be split into chunks and stored in ChromaDB

## Example
```python
from src.document_loader import DocumentLoader

loader = DocumentLoader("data/")
documents = loader.load_documents()
```

## Notes
- Documents are processed recursively (including subdirectories)
- Default chunk size is 1000 characters with 200 overlap
- Metadata includes filename and type
