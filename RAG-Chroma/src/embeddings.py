"""
Embeddings management module
Supports Sentence-Transformers, OpenAI, and HuggingFace models
"""

import os
from typing import List, Optional, Union, Callable
from abc import ABC, abstractmethod
import numpy as np

# Lazy imports para evitar dependencias innecesarias
_sentence_transformer = None
_openai_client = None


def get_sentence_transformer():
    """Lazy import de SentenceTransformer"""
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer
    return _sentence_transformer


class BaseEmbeddingManager(ABC):
    """Abstract base class for embedding managers"""
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        pass
    
    @abstractmethod
    def get_embedding_function(self):
        """Return embedding function compatible with ChromaDB"""
        pass


class EmbeddingManager(BaseEmbeddingManager):
    """Gestor de embeddings usando Sentence-Transformers"""
    
    # Modelos recomendados
    RECOMMENDED_MODELS = {
        'small': 'all-MiniLM-L6-v2',
        'medium': 'all-mpnet-base-v2', 
        'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
        'spanish': 'hiiamsid/sentence_similarity_spanish_es'
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the embedding manager
        
        Args:
            model_name: Name of the embedding model
            device: Device (cpu/cuda), None for auto
            cache_folder: Folder for model cache
        """
        # Clean prefix if exists
        if model_name.startswith("sentence-transformers/"):
            model_name = model_name.replace("sentence-transformers/", "")
        
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        
        print(f"🔄 Loading embedding model: {model_name}")
        
        SentenceTransformer = get_sentence_transformer()
        self.model = SentenceTransformer(
            model_name, 
            device=device,
            cache_folder=cache_folder
        )
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded - Dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts
            batch_size: Batch size
            show_progress_bar: Show progress bar
            normalize: Normalize embeddings (L2)
            
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a query
        
        Args:
            query: Query text
            normalize: Normalize embedding
            
        Returns:
            Query embedding
        """
        return self.model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
    
    def get_embedding_function(self):
        """
        Return embedding function for LangChain/ChromaDB
        
        Returns:
            Embedding function compatible with LangChain
        """
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device or 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, dot, euclidean)
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            from numpy.linalg import norm
            cos_sim = np.dot(embedding1, embedding2) / (
                norm(embedding1) * norm(embedding2) + 1e-8
            )
            return float(cos_sim)
        elif metric == "dot":
            return float(np.dot(embedding1, embedding2))
        elif metric == "euclidean":
            return float(-np.linalg.norm(embedding1 - embedding2))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarity between query and multiple documents
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Document embeddings matrix
            
        Returns:
            Array of similarity scores
        """
        # Normalize if not already normalized
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = document_embeddings / (
            np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = np.dot(doc_norms, query_norm)
        return similarities


class OpenAIEmbeddingManager(BaseEmbeddingManager):
    """Embedding manager using OpenAI API"""
    
    AVAILABLE_MODELS = {
        'small': 'text-embedding-3-small',
        'large': 'text-embedding-3-large',
        'ada': 'text-embedding-ada-002'
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize with OpenAI
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Embedding model
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Dimensions per model
        self.embedding_dim = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }.get(model, 1536)
        
        print(f"✓ OpenAI Embeddings initialized - Model: {model}")
    
    def encode(
        self, 
        texts: List[str],
        batch_size: int = 100,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings with OpenAI"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        response = self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def get_embedding_function(self):
        """Return embedding function for LangChain"""
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.model
        )


def create_embedding_manager(
    provider: str = "sentence-transformers",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingManager:
    """
    Factory function to create the appropriate embedding manager
    
    Args:
        provider: Embedding provider (sentence-transformers, openai)
        model_name: Model name
        **kwargs: Additional arguments
        
    Returns:
        Embedding manager instance
    """
    if provider == "sentence-transformers":
        model = model_name or "all-MiniLM-L6-v2"
        return EmbeddingManager(model_name=model, **kwargs)
    elif provider == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddingManager(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
