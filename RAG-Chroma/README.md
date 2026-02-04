# 🔍 RAG with ChromaDB

A **Retrieval-Augmented Generation (RAG)** system using ChromaDB as vector database. Designed to run on **Google Colab** with a **100% FREE and UNLIMITED** local LLM.

![RAG Architecture](https://miro.medium.com/v2/resize:fit:1400/1*3q6xmUkB4l5VJv8Q8a8OVA.png)

## 🚀 Quick Start on Colab

1. **Open the notebook in Colab:**
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/RAG-Chroma/blob/main/RAG_ChromaDB_Colab.ipynb)

2. **Enable GPU:**
   - Go to `Runtime > Change runtime type > T4 GPU`

3. **Run the notebook cells in order**

> ⚡ **No API key needed!** The model runs locally on Colab's free GPU.

## 📁 Project Structure

```
RAG-Chroma/
├── 📓 RAG_ChromaDB_Colab.ipynb  # Main notebook
├── 📁 data/                      # Documents to index
│   ├── machine_learning.txt      # ML fundamentals
│   ├── deep_learning.txt         # Neural networks
│   ├── transformers.txt          # Transformer architecture
│   └── rag_systems.txt           # RAG systems
├── 📄 requirements.txt           # Dependencies
└── 📄 README.md                  # This file
```

## 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| Vector Store | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Orchestration | LangChain |
| LLM | TinyLlama 1.1B (Local, FREE, Unlimited) |
| Document Loaders | PyPDF, python-docx |

## 📊 Features

- ✅ **Multiple formats**: TXT, PDF, DOCX
- ✅ **Smart chunking**: Recursive character splitting
- ✅ **Free embeddings**: Sentence Transformers (local)
- ✅ **FREE & Unlimited LLM**: Runs locally on GPU
- ✅ **Persistence**: ChromaDB persists to disk
- ✅ **Source citation**: Shows where information comes from
- ✅ **Interactive chat**: Q&A interface in notebook

## 🎯 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      INDEXING (Offline)                      │
├─────────────────────────────────────────────────────────────┤
│  Documents → Chunking → Embeddings → ChromaDB               │
│     📄          ✂️          🔢           💾                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      QUERY (Online)                          │
├─────────────────────────────────────────────────────────────┤
│  Question → Embedding → Search → Context → LLM → Answer      │
│     ❓          🔢         🔍        📚      🤖      💬       │
└─────────────────────────────────────────────────────────────┘
```

## 📝 Add Your Own Documents

### Option 1: Upload in Colab
Run the "Upload files" cell and select your documents.

### Option 2: Local data/ folder
Place your files in the `data/` folder before uploading to Colab:
- `.txt` - Text files
- `.pdf` - PDF documents
- `.docx` - Word documents

### Option 3: Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
# Then use: load_documents('/content/drive/MyDrive/my_documents')
```

## ⚙️ Configuration

### Adjust Chunking
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Increase for longer documents
    chunk_overlap=50,    # Overlap between chunks
)
```

### Change Number of Retrieved Documents
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top 5 documents
)
```

## 🔬 RAG Pipeline Explained

1. **Document Loading**: Load documents from various formats (TXT, PDF, DOCX)
2. **Text Splitting**: Divide documents into manageable chunks
3. **Embedding Generation**: Convert text chunks to vector representations
4. **Vector Storage**: Store embeddings in ChromaDB
5. **Semantic Search**: Find relevant chunks based on query similarity
6. **Answer Generation**: Use LLM to generate answers from retrieved context

## 📈 Performance Tips

- Enable GPU for faster inference: `Runtime > Change runtime type > T4 GPU`
- Reduce `k` in retriever for faster responses
- Adjust `chunk_size` based on your document types
- Use smaller embedding models for faster indexing

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

MIT License - Feel free to use this project for learning and development.

---

**Built with ❤️ using LangChain, ChromaDB, and Hugging Face Transformers**
