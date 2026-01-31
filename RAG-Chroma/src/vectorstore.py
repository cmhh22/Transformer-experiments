"""
ChromaDB management module
CRUD operations on vector store
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class ChromaManager:
    """ChromaDB manager for vector storage"""
    
    def __init__(
        self,
        persist_directory: str = "./models/chroma_db",
        collection_name: str = "rag_collection",
        embedding_function: Any = None
    ):
        """
        Initialize ChromaDB
        
        Args:
            persist_directory: Persistence directory
            collection_name: Collection name
            embedding_function: Embedding function
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent ChromaDB client (updated API)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )
        print(f"✓ Collection '{collection_name}' ready ({self.collection.count()} documents)")
        
        # Create LangChain vectorstore
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to vectorstore
        
        Args:
            documents: List of documents
            batch_size: Batch size for insertion
        """
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to ChromaDB...")
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            print(f"Batch {i//batch_size + 1}: {len(batch)} documents added")
        
        # Persist changes
        self.persist()
        print("✓ Documents added successfully")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add texts directly
        
        Args:
            texts: List of texts
            metadatas: Optional metadata for each text
            
        Returns:
            List of generated IDs
        """
        ids = self.vectorstore.add_texts(texts, metadatas=metadatas)
        self.persist()
        return ids
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Similarity search
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filters
            
        Returns:
            List of relevant documents
        """
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
        return results
    
    def search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Search with similarity scores
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filters
            
        Returns:
            List of tuples (document, score)
        """
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
        return results
    
    def delete_collection(self) -> None:
        """Delete entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' deleted")
    
    def persist(self) -> None:
        """Persist changes to disk (automatic in PersistentClient)"""
        # In ChromaDB >= 0.4.x with PersistentClient, persistence is automatic
        pass
    
    def get_collection_info(self) -> Dict:
        """
        Get collection information
        
        Returns:
            Dictionary with information
        """
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"}
        )
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"✓ Collection '{self.collection_name}' cleared")
