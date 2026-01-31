"""
Response generation module
Combines retrieved context with LLMs
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


class ResponseGenerator:
    """Response generator using RAG"""
    
    def __init__(
        self,
        retriever,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        use_local_model: bool = True
    ):
        """
        Initialize generator
        
        Args:
            retriever: RetrieverManager instance
            model_name: LLM model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            use_local_model: Use local model (True) or OpenAI API (False)
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_model = use_local_model
        self.llm = None
        
        # Prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Initialize LLM if API key available
        if not use_local_model and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                print(f"✓ Generador configurado con OpenAI: {model_name}")
            except ImportError:
                print("⚠ langchain-openai not installed, using local generation")
        else:
            print(f"✓ Generator configured (local mode)")
    
    def generate(
        self,
        query: str,
        documents: Optional[List[Document]] = None,
        k: Optional[int] = None
    ) -> str:
        """
        Generate response based on context
        
        Args:
            query: User question
            documents: Context documents (optional)
            k: Number of documents to retrieve if not provided
            
        Returns:
            Generated response
        """
        # Retrieve documents if not provided
        if documents is None:
            documents = self.retriever.retrieve(query, k=k)
        
        if not documents:
            return "No relevant information found to answer the question."
        
        # Build context
        context = self.retriever.get_context_string(documents)
        
        # Generate response (simulated)
        # In production, use OpenAI API or local model
        response = self._generate_with_template(context, query)
        
        return response
    
    def _generate_with_template(self, context: str, question: str) -> str:
        """
        Generate response using template
        
        Args:
            context: Retrieved context
            question: Question
            
        Returns:
            Generated response
        """
        # Simplified version (placeholder)
        # In production, integrate with OpenAI, Anthropic, or local model
        
        prompt = self.prompt_template.format(
            context=context[:2000],  # Limit context
            question=question
        )
        
        # Si hay LLM configurado, usarlo
        if self.llm:
            try:
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"⚠ Error with LLM: {e}")
        
        # Local context-based generation (extractive)
        response = self._extract_relevant_answer(context, question)
        return response
    
    def _extract_relevant_answer(self, context: str, question: str) -> str:
        """
        Extract relevant answer from context without LLM
        
        Args:
            context: Document context
            question: User question
            
        Returns:
            Answer extracted from context
        """
        # Find relevant sentences
        question_terms = set(question.lower().split())
        sentences = context.replace('\n', ' ').split('.')
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Ignore very short sentences
                sentence_terms = set(sentence.lower().split())
                score = len(question_terms & sentence_terms)
                if score > 0:
                    scored_sentences.append((score, sentence))
        
        # Sort by relevance
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Build response
        if scored_sentences:
            top_sentences = [s[1] for s in scored_sentences[:3]]
            response = "Based on the documents:\n\n"
            response += ". ".join(top_sentences) + "."
            return response
        
        return "No specific information found in the provided documents."
    
    def generate_with_citations(
        self,
        query: str,
        k: Optional[int] = None
    ) -> dict:
        """
        Generate response with source citations
        
        Args:
            query: User question
            k: Number of documents
            
        Returns:
            Dict with answer and sources
        """
        # Retrieve documents with scores
        docs_with_scores = self.retriever.retrieve_with_scores(query, k=k)
        
        if not docs_with_scores:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }
        
        documents = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Generate response
        answer = self.generate(query, documents)
        
        # Prepare sources
        sources = []
        for doc, score in zip(documents, scores):
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
                "content": doc.page_content[:200] + "..."
            })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def stream_generate(self, query: str, k: Optional[int] = None):
        """
        Generate response in streaming (for interfaces)
        
        Args:
            query: User question
            k: Number of documents
            
        Yields:
            Response chunks
        """
        # Retrieve documents
        documents = self.retriever.retrieve(query, k=k)
        
        if not documents:
            yield "No relevant information found."
            return
        
        # Generate response
        response = self.generate(query, documents)
        
        # Simulate streaming
        for word in response.split():
            yield word + " "
