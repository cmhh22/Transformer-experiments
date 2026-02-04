# Transformer Experiments

Educational projects experimenting with modern Transformer models. The goal is to learn by doing: from fine-tuning classifiers to building complete RAG systems.

## 📚 Projects

### 🔬 BERT vs Qwen: Emotion Classification

**What I learned:**
- Fine-tuning bidirectional transformers (BERT) vs decoders (Qwen)
- Quantitative comparison: BERT is 6x faster, Qwen is 0.14% more accurate
- Using HuggingFace Transformers, mixed precision training (FP16)
- Interpretability: confusion matrices, per-class metrics

**Technologies:** PyTorch, HuggingFace Transformers, Emotion Dataset (6 classes)

**Result:** 92.6% accuracy with both models on emotion classification

[View full project →](./BERT-Qwen-Classification)

---

### 🔍 RAG with ChromaDB

**What I learned:**
- RAG architecture: retrieval + generation for Q&A on documents
- Vector databases: ChromaDB for semantic search
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2) for dense representations
- LangChain to orchestrate the complete pipeline
- Local LLMs: TinyLlama 1.1B, Groq, Google Gemini (free APIs)

**Technologies:** ChromaDB, LangChain, Sentence Transformers, TinyLlama

**Result:** Functional RAG system that answers questions citing sources

[View full project →](./RAG-Chroma)

## 🎯 Concepts Explored

- **Transfer Learning**: Adapting pre-trained models to specific tasks
- **Fine-tuning Strategies**: Full learning vs upper layers only
- **Embeddings**: Vector representations for semantic search
- **RAG Architecture**: Combining retrieval with generation for grounded answers
- **Model Comparison**: Systematic benchmarking (speed, accuracy, trade-offs)

## 📚 Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 👤 Author

**Carlos Manuel Hernández**
- GitHub: [@cmhh22](https://github.com/cmhh22)


## 📄 License

MIT License - see [LICENSE](LICENSE)
