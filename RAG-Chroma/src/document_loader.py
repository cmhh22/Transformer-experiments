"""
Document loading module for RAG
Supports multiple formats: PDF, TXT, MD, DOCX
"""

import os
from pathlib import Path
from typing import List, Union, Optional

# LangChain imports actualizados
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """Document loader with intelligent chunking"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.markdown'}
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding: str = "utf-8"
    ):
        """
        Initialize the document loader
        
        Args:
            data_dir: Directory with documents
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            encoding: Text file encoding
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = encoding
        
        # Configure text splitter with optimized separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        # Extension to loader mapping
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': self._create_text_loader,
            '.md': self._create_markdown_loader,
            '.markdown': self._create_markdown_loader,
            '.docx': Docx2txtLoader
        }
    
    def _create_text_loader(self, file_path: str):
        """Create TextLoader with configured encoding"""
        return TextLoader(file_path, encoding=self.encoding)
    
    def _create_markdown_loader(self, file_path: str):
        """Create loader for Markdown files"""
        try:
            return UnstructuredMarkdownLoader(file_path)
        except Exception:
            # Fallback to TextLoader if it fails
            return TextLoader(file_path, encoding=self.encoding)
    
    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of documents (chunks)
        """
        ext = file_path.suffix.lower()
        
        if ext not in self.loaders:
            print(f"⚠ Unsupported format: {ext} - Skipping {file_path.name}")
            return []
        
        try:
            loader_factory = self.loaders[ext]
            
            # Create loader (some are functions, others are classes)
            if callable(loader_factory) and ext in ['.txt', '.md', '.markdown']:
                loader = loader_factory(str(file_path))
            else:
                loader = loader_factory(str(file_path))
            
            documents = loader.load()
            
            # Apply chunking
            chunks = self.text_splitter.split_documents(documents)
            
            # Add enriched metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': str(file_path.name),
                    'file_path': str(file_path),
                    'file_type': ext,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
            
            return chunks
            
        except UnicodeDecodeError:
            print(f"⚠ Encoding error in {file_path.name}, trying latin-1...")
            try:
                loader = TextLoader(str(file_path), encoding='latin-1')
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'source': str(file_path.name),
                        'file_path': str(file_path),
                        'file_type': ext,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })
                return chunks
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
                return []
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            return []
    
    def load_documents(self, file_pattern: Optional[str] = None) -> List[Document]:
        """
        Load all documents from the directory
        
        Args:
            file_pattern: Optional file pattern (e.g., "*.txt")
            
        Returns:
            List of all documents (chunks)
        """
        if not self.data_dir.exists():
            print(f"✗ Directory not found: {self.data_dir}")
            return []
        
        all_documents = []
        files_processed = 0
        files_failed = 0
        
        # Search for files recursively
        pattern = file_pattern or "*"
        for file_path in self.data_dir.rglob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Ignore README files
                if file_path.name.lower().startswith('readme'):
                    continue
                    
                print(f"  📄 Processing: {file_path.name}")
                docs = self.load_document(file_path)
                if docs:
                    all_documents.extend(docs)
                    files_processed += 1
                else:
                    files_failed += 1
        
        print(f"\n📊 Load Summary:")
        print(f"   - Files processed: {files_processed}")
        print(f"   - Files failed: {files_failed}")
        print(f"   - Total chunks generated: {len(all_documents)}")
        
        return all_documents
    
    def load_from_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load text directly (no file)
        
        Args:
            text: Text to process
            metadata: Optional metadata
            
        Returns:
            List of documents (chunks)
        """
        doc = Document(page_content=text, metadata=metadata or {})
        chunks = self.text_splitter.split_documents([doc])
        
        # Add indices to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source': 'direct_text'
            })
        
        return chunks
    
    def get_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics of loaded documents
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": total_chars / len(documents),
            "unique_sources": len(sources),
            "sources": list(sources)
        }
