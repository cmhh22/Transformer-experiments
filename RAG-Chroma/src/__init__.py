"""RAG-Chroma package initialization"""

__version__ = "0.1.0"
__author__ = "Carlos M. Hernández"

from .document_loader import DocumentLoader
from .embeddings import EmbeddingManager
from .vectorstore import ChromaManager
from .retriever import RetrieverManager
from .generator import ResponseGenerator
from .evaluator import RAGEvaluator

__all__ = [
    "DocumentLoader",
    "EmbeddingManager",
    "ChromaManager",
    "RetrieverManager",
    "ResponseGenerator",
    "RAGEvaluator"
]
