"""
RAG-Chroma: Retrieval-Augmented Generation System
Complete implementation with ChromaDB and LangChain
"""

import os
from pathlib import Path
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vectorstore import ChromaManager
from src.retriever import RetrieverManager
from src.generator import ResponseGenerator
from src.evaluator import RAGEvaluator


def main():
    """Complete RAG pipeline"""
    
    print("=" * 60)
    print("RAG-Chroma: Retrieval-Augmented Generation System")
    print("=" * 60)
    
    # Path configuration
    data_dir = Path("data")
    models_dir = Path("models")
    chroma_dir = models_dir / "chroma_db"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # 1. Load documents
    print("\n[1/6] Loading documents...")
    loader = DocumentLoader(data_dir)
    documents = loader.load_documents()
    print(f"✓ Documents loaded: {len(documents)}")
    
    # 2. Generate embeddings
    print("\n[2/6] Generating embeddings...")
    embedding_manager = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"✓ Embedding model: {embedding_manager.model_name}")
    
    # 3. Create/Update ChromaDB
    print("\n[3/6] Initializing ChromaDB...")
    chroma_manager = ChromaManager(
        persist_directory=str(chroma_dir),
        embedding_function=embedding_manager.get_embedding_function()
    )
    
    # Add documents to vectorstore
    if documents:
        chroma_manager.add_documents(documents)
        print(f"✓ Documents added to ChromaDB")
    
    # 4. Configure retriever
    print("\n[4/6] Configuring retriever...")
    retriever = RetrieverManager(
        vectorstore=chroma_manager.vectorstore,
        k=3
    )
    print(f"✓ Retriever configured (k={retriever.k})")
    
    # 5. Configure generator
    print("\n[5/6] Configuring response generator...")
    generator = ResponseGenerator(
        retriever=retriever,
        model_name="gpt-3.5-turbo"  # Change as needed
    )
    print(f"✓ Generator configured")
    
    # 6. Usage example
    print("\n[6/6] Running example query...")
    query = "What is the main content of the documents?"
    
    # Retrieve relevant documents
    relevant_docs = retriever.retrieve(query)
    print(f"✓ Documents retrieved: {len(relevant_docs)}")
    
    # Generate response
    response = generator.generate(query, relevant_docs)
    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)
    print("RESPONSE:", response)
    print("=" * 60)
    
    # 7. Evaluation (optional)
    print("\n[Evaluation] Running RAGAS metrics...")
    evaluator = RAGEvaluator()
    
    # Evaluation example
    eval_results = evaluator.evaluate(
        query=query,
        contexts=[doc.page_content for doc in relevant_docs],
        answer=response
    )
    
    print("\nEvaluation metrics:")
    for metric, score in eval_results.items():
        print(f"  - {metric}: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
