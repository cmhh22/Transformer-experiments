# 🔍 RAG-Chroma: Retrieval-Augmented Generation with ChromaDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USER/RAG-Chroma/blob/main/notebooks/rag_colab.ipynb)

## 📋 Description
A complete **Retrieval-Augmented Generation (RAG)** system using ChromaDB as a vector database. This project implements a full pipeline for embeddings, semantic search, and contextualized response generation using Hugging Face models.

## 🎯 Objectives
- ✅ Implement a functional RAG system with ChromaDB
- ✅ Experiment with embedding models (Sentence-Transformers)
- ✅ Optimize relevant document retrieval
- ✅ Generate contextualized responses with FLAN-T5

## 🛠️ Technologies
| Component | Technology |
|-----------|------------|
| **Vector Store** | ChromaDB |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Generator** | FLAN-T5 (google/flan-t5-base) |
| **Framework** | PyTorch, Transformers |

## 📊 Features
- ✅ AI/ML knowledge base (10+ documents)
- ✅ Vector embeddings with Sentence Transformers
- ✅ Semantic search with cosine similarity
- ✅ Metadata filtering (category, topic)
- ✅ Contextual response generation
- ✅ Evaluation with MRR and Accuracy metrics
- ✅ **100% self-contained for Google Colab**

## 🚀 Quick Start (Google Colab)

**The easiest way to try the project:**

1. Open the notebook in Colab: [rag_colab.ipynb](notebooks/rag_colab.ipynb)
2. Run all cells in order
3. Ask your own questions!

## 💻 Local Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Interactive Notebook (recommended)
```bash
jupyter notebook notebooks/rag_colab.ipynb
```

### Main Pipeline
```bash
python main.py
```

### Functionality Tests
```bash
python test_quick.py
```

## 📁 Project Structure
```
RAG-Chroma/
├── main.py                 # Main script
├── test_quick.py           # Quick tests
├── requirements.txt        # Dependencies
├── README.md
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading
│   ├── embeddings.py       # Embedding generation
│   ├── vectorstore.py      # ChromaDB operations
│   ├── retriever.py        # Document retrieval
│   ├── generator.py        # Response generation
│   └── evaluator.py        # RAGAS metrics
├── data/                   # Source documents
│   └── sample_document.txt # Sample document
├── models/                 # ChromaDB and models
│   └── chroma_db/          # Vector database
└── notebooks/              # Experimentation
    └── rag_colab.ipynb     # Interactive demo
```

## 🔧 Configuration

### Add Documents
Place files in `data/` (supports .txt, .pdf, .md, .docx)

### Change Embedding Model
```python
from src.embeddings import EmbeddingManager

# Available options:
manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")      # Lightweight
manager = EmbeddingManager(model_name="all-mpnet-base-v2")     # Accurate
manager = EmbeddingManager(model_name="paraphrase-multilingual-MiniLM-L12-v2")  # Multilingual
```

### Use OpenAI for Generation
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

from src.generator import ResponseGenerator
generator = ResponseGenerator(retriever=retriever, use_local_model=False)
```

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Context Relevance | Relevance of retrieved context |
| Answer Relevance | Relevance of generated answer |
| Faithfulness | Answer fidelity to context |
| Answer Correctness | Correctness vs ground truth |

## 🔬 Main Components

### DocumentLoader
```python
loader = DocumentLoader(data_dir="data", chunk_size=1000, chunk_overlap=200)
documents = loader.load_documents()
```

### ChromaManager
```python
chroma = ChromaManager(persist_directory="./models/chroma_db", embedding_function=ef)
chroma.add_documents(documents)
```

### RetrieverManager
```python
retriever = RetrieverManager(vectorstore=chroma.vectorstore, k=4)
docs = retriever.retrieve("What is RAG?")
docs_diverse = retriever.mmr_retrieve("What is RAG?", lambda_mult=0.5)
```

### ResponseGenerator
```python
generator = ResponseGenerator(retriever=retriever)
response = generator.generate("What is machine learning?")
result = generator.generate_with_citations("question", k=3)
```

## 📝 Notes
- Part of transformer-experiments project
- Updated: January 2026
- Compatible with ChromaDB >= 0.4.x and LangChain >= 0.2.x

## 📄 License
MIT License

## 👤 Author
Carlos M. Hernández
