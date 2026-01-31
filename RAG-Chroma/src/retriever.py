"""
Document retrieval module
Implements different retrieval strategies
"""

from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class RetrieverManager:
    """Document retrieval manager"""
    
    def __init__(
        self,
        vectorstore: VectorStore,
        k: int = 4,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None
    ):
        """
        Initialize retriever
        
        Args:
            vectorstore: Vector store to use
            k: Number of documents to retrieve
            search_type: Search type (similarity, mmr)
            score_threshold: Minimum score threshold
        """
        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        
        # Configure base retriever
        self.retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            } if score_threshold else {"k": k}
        )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            k: Number of documents (override)
            filter: Metadata filters
            
        Returns:
            List of relevant documents
        """
        k = k or self.k
        
        if filter:
            # Search with filter
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter
            )
        else:
            # Use configured retriever
            results = self.retriever.invoke(query)[:k]
        
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        Retrieve documents with scores
        
        Args:
            query: Search query
            k: Number of documents
            
        Returns:
            List of tuples (document, score)
        """
        k = k or self.k
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Filter by threshold if configured
        if self.score_threshold:
            results = [
                (doc, score) for doc, score in results
                if score >= self.score_threshold
            ]
        
        return results
    
    def mmr_retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Retrieval with Maximal Marginal Relevance (diversity)
        
        Args:
            query: Search query
            k: Number of final documents
            fetch_k: Number of documents to fetch initially
            lambda_mult: Relevance-diversity balance (0-1)
            
        Returns:
            List of diverse and relevant documents
        """
        k = k or self.k
        
        results = self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        return results
    
    def hybrid_retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Hybrid retrieval (combining multiple strategies)
        
        Args:
            query: Search query
            k: Number of documents
            alpha: Weight of similarity search (1-alpha for MMR)
            
        Returns:
            List of documents
        """
        k = k or self.k
        k_split = max(k // 2, 1)
        
        # Get similarity search results
        similarity_results = self.retrieve(query, k=k_split)
        
        # Get MMR results
        mmr_results = self.mmr_retrieve(query, k=k_split)
        
        # Combine and deduplicate
        seen_ids = set()
        combined = []
        
        # Alternate between both lists
        for doc1, doc2 in zip(similarity_results, mmr_results):
            doc1_id = id(doc1.page_content)
            doc2_id = id(doc2.page_content)
            
            if doc1_id not in seen_ids:
                combined.append(doc1)
                seen_ids.add(doc1_id)
            
            if doc2_id not in seen_ids:
                combined.append(doc2)
                seen_ids.add(doc2_id)
            
            if len(combined) >= k:
                break
        
        return combined[:k]
    
    def get_context_string(self, documents: List[Document]) -> str:
        """
        Convert document list to context string
        
        Args:
            documents: List of documents
            
        Returns:
            String with all context concatenated
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'unknown')
            context_parts.append(
                f"[Document {i} - {source}]\n{doc.page_content}\n"
            )
        return "\n".join(context_parts)
