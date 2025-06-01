"""
Document processing module for RAG pipeline.
Handles document loading, chunking, and preprocessing.
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging


@dataclass
class Document:
    """
    Document class for storing text content and metadata.
    """
    content: str
    metadata: Dict[str, Any] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize document ID if not provided."""
        if self.id is None:
            # Generate ID based on content hash
            self.id = hashlib.md5(self.content.encode()).hexdigest()
            
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """
    Processor for loading, chunking, and preprocessing documents.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata_fields: List[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each document chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            metadata_fields: List of metadata fields to extract
            logger: Logger instance
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_fields = metadata_fields or []
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
    def load_text(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """
        Load text content into a Document.
        
        Args:
            text: Text content
            metadata: Optional metadata
            
        Returns:
            Document object
        """
        return Document(content=text, metadata=metadata or {})
    
    def load_file(self, file_path: str) -> Document:
        """
        Load document from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Extract file metadata
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_type": os.path.splitext(file_path)[1].lower(),
            "file_size": os.path.getsize(file_path)
        }
        
        # Read file content based on file type
        file_type = metadata["file_type"]
        
        if file_type in [".txt", ".md", ".py", ".java", ".js", ".html", ".css", ".json"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
        elif file_type == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert JSON to string representation
                content = json.dumps(data, indent=2)
                
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        return Document(content=content, metadata=metadata)
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        chunks = []
        
        # Simple chunking by character count
        # In a production system, you would use more sophisticated chunking
        # that respects sentence and paragraph boundaries
        
        if len(content) <= self.chunk_size:
            # Document is small enough, no need to chunk
            return [document]
            
        # Chunk the document
        start = 0
        chunk_id = 0
        
        while start < len(content):
            # Calculate end position
            end = min(start + self.chunk_size, len(content))
            
            # If not at the end of the document, try to break at a sentence boundary
            if end < len(content):
                # Look for sentence boundaries (., !, ?)
                sentence_end = max(
                    content.rfind(". ", start, end),
                    content.rfind("! ", start, end),
                    content.rfind("? ", start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1  # Include the period
            
            # Extract chunk text
            chunk_text = content[start:end].strip()
            
            # Create chunk metadata
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_start": start,
                "chunk_end": end,
                "parent_id": document.id
            })
            
            # Create chunk document
            chunk = Document(
                content=chunk_text,
                metadata=chunk_metadata,
                id=f"{document.id}_chunk_{chunk_id}"
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
                
            chunk_id += 1
            
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text content.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs (simplified)
        text = re.sub(r'https?://\S+', '[URL]', text)
        
        return text
    
    def process_document(self, document: Document) -> List[Document]:
        """
        Process a document: preprocess and chunk.
        
        Args:
            document: Document to process
            
        Returns:
            List of processed document chunks
        """
        # Preprocess document content
        document.content = self.preprocess_text(document.content)
        
        # Chunk document
        chunks = self.chunk_document(document)
        
        self.logger.info(f"Processed document {document.id} into {len(chunks)} chunks")
        
        return chunks
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process multiple documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed document chunks
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
            
        return all_chunks
