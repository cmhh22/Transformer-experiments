"""
Quick test to verify installation and basic functionality of RAG-Chroma
"""

import sys
import shutil
from pathlib import Path


def test_imports():
    """Verify that all dependencies are installed"""
    print("Checking imports...")
    
    try:
        import chromadb
        print("✓ chromadb")
    except ImportError as e:
        print(f"✗ chromadb: {e}")
        return False
    
    try:
        import langchain
        print("✓ langchain")
    except ImportError as e:
        print(f"✗ langchain: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers")
    except ImportError as e:
        print(f"✗ sentence-transformers: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    return True


def test_chromadb():
    """Basic ChromaDB test"""
    print("\nTesting ChromaDB...")
    
    try:
        import chromadb
        
        # Create in-memory client (updated API)
        client = chromadb.Client()
        
        # Create test collection
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"description": "Test collection"}
        )
        
        # Add test documents
        collection.add(
            documents=["Test document 1", "Test document 2"],
            ids=["doc1", "doc2"]
        )
        
        # Query
        results = collection.query(
            query_texts=["test"],
            n_results=2
        )
        
        print(f"✓ ChromaDB working correctly")
        print(f"  Documents found: {len(results['documents'][0])}")
        
        # Cleanup
        client.delete_collection("test_collection")
        return True
        
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """Basic embeddings test"""
    print("\nTesting embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load small model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate test embedding
        sentences = ["This is a test", "Another example sentence"]
        embeddings = model.encode(sentences)
        
        print(f"✓ Embeddings generated correctly")
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Vectors generated: {embeddings.shape[0]}")
        return True
        
    except Exception as e:
        print(f"✗ Embeddings error: {e}")
        return False


def test_structure():
    """Verify directory structure"""
    print("\nVerifying directory structure...")
    
    required_dirs = ["src", "data", "models", "notebooks"]
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (does not exist)")
            all_exist = False
    
    return all_exist


def test_rag_pipeline():
    """Complete RAG pipeline test"""
    print("\nTesting complete RAG pipeline...")
    
    try:
        from src.document_loader import DocumentLoader
        from src.embeddings import EmbeddingManager
        from src.vectorstore import ChromaManager
        from src.retriever import RetrieverManager
        from src.generator import ResponseGenerator
        from src.evaluator import RAGEvaluator
        
        # Setup
        test_dir = Path("models/chroma_test_pipeline")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        # 1. Load documents
        loader = DocumentLoader("data", chunk_size=500)
        documents = loader.load_documents()
        print(f"  ✓ Documents loaded: {len(documents)}")
        
        # 2. Create embeddings
        embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        print(f"  ✓ Embedding model: {embedding_manager.embedding_dim}D")
        
        # 3. ChromaDB
        chroma = ChromaManager(
            persist_directory=str(test_dir),
            embedding_function=embedding_manager.get_embedding_function()
        )
        chroma.add_documents(documents)
        print(f"  ✓ VectorStore: {chroma.get_collection_info()['count']} docs")
        
        # 4. Retriever
        retriever = RetrieverManager(vectorstore=chroma.vectorstore, k=3)
        query = "What is RAG?"
        results = retriever.retrieve(query)
        print(f"  ✓ Retrieval: {len(results)} documents for '{query}'")
        
        # 5. Generator
        generator = ResponseGenerator(retriever=retriever, use_local_model=True)
        response = generator.generate(query, results)
        print(f"  ✓ Generation: {len(response)} characters")
        
        # 6. Evaluator
        evaluator = RAGEvaluator()
        eval_results = evaluator.evaluate(
            query=query,
            contexts=[doc.page_content for doc in results],
            answer=response
        )
        print(f"  ✓ Evaluation: {len(eval_results)} metrics")
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("RAG-Chroma - Quick Verification Test")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Structure": test_structure(),
        "ChromaDB": test_chromadb(),
        "Embeddings": test_embeddings(),
        "RAG Pipeline": test_rag_pipeline()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed successfully")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
