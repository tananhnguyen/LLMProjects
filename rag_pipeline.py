"""
RAG pipeline module for Retrieval-Augmented Generation.
Integrates document processing, embedding generation, and vector search.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from src.rag.document_processor import Document, DocumentProcessor
from src.rag.embedding_generator import EmbeddingGenerator
from src.rag.vector_database import VectorDatabase


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Combines document processing, embedding generation, and vector search.
    """
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessor] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_database: Optional[VectorDatabase] = None,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            document_processor: Document processor instance
            embedding_generator: Embedding generator instance
            vector_database: Vector database instance
            top_k: Number of documents to retrieve
            logger: Logger instance
        """
        # Initialize components with defaults if not provided
        self.document_processor = document_processor or DocumentProcessor()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_database = vector_database or VectorDatabase(
            dimension=384  # Default for all-MiniLM-L6-v2
        )
        
        self.top_k = top_k
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Initialized RAG pipeline")
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts to the RAG pipeline.
        
        Args:
            texts: List of text content
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")
            
        # Process documents
        documents = []
        for text, metadata in zip(texts, metadatas):
            doc = self.document_processor.load_text(text, metadata)
            documents.append(doc)
            
        # Chunk documents
        chunked_docs = []
        for doc in documents:
            chunks = self.document_processor.process_document(doc)
            chunked_docs.extend(chunks)
            
        # Generate embeddings
        embedded_docs = self.embedding_generator.embed_documents(chunked_docs)
        
        # Add to vector database
        doc_ids = self.vector_database.add_documents(embedded_docs)
        
        self.logger.info(f"Added {len(doc_ids)} documents to RAG pipeline")
        
        return doc_ids
        
    def add_files(self, file_paths: List[str]) -> List[str]:
        """
        Add files to the RAG pipeline.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of document IDs
        """
        # Process documents
        documents = []
        for file_path in file_paths:
            try:
                doc = self.document_processor.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                self.logger.error(f"Error loading file {file_path}: {e}")
                
        # Chunk documents
        chunked_docs = []
        for doc in documents:
            chunks = self.document_processor.process_document(doc)
            chunked_docs.extend(chunks)
            
        # Generate embeddings
        embedded_docs = self.embedding_generator.embed_documents(chunked_docs)
        
        # Add to vector database
        doc_ids = self.vector_database.add_documents(embedded_docs)
        
        self.logger.info(f"Added {len(doc_ids)} document chunks from {len(file_paths)} files to RAG pipeline")
        
        return doc_ids
        
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve (overrides instance default)
            filter_dict: Optional metadata filter
            
        Returns:
            List of retrieved documents with scores
        """
        # Use instance default if not specified
        if top_k is None:
            top_k = self.top_k
            
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector database
        results = self.vector_database.search(
            query_embedding,
            k=top_k,
            filter_dict=filter_dict
        )
        
        self.logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        
        return results
        
    def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        format_template: str = "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant documents and format them for LLM input.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            filter_dict: Optional metadata filter
            format_template: Template for formatting the context and query
            
        Returns:
            Formatted prompt and list of retrieved documents
        """
        # Retrieve documents
        results = self.retrieve(query, top_k, filter_dict)
        
        # Extract content from results
        contexts = [result["content"] for result in results]
        
        # Join contexts
        context_text = "\n\n".join(contexts)
        
        # Format prompt
        formatted_prompt = format_template.format(
            context=context_text,
            query=query
        )
        
        return formatted_prompt, results
        
    def save(self, directory: str = "data/rag_pipeline") -> Dict[str, str]:
        """
        Save the RAG pipeline components.
        
        Args:
            directory: Directory to save components
            
        Returns:
            Dictionary of saved file paths
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save vector database
        vector_db_path = self.vector_database.save(prefix="rag_pipeline")
        
        # Save configuration
        config = {
            "top_k": self.top_k,
            "vector_db_prefix": "rag_pipeline",
            "embedding_model": self.embedding_generator.model_name,
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap
        }
        
        import json
        config_path = os.path.join(directory, "rag_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Saved RAG pipeline to {directory}")
        
        return {
            "vector_db": vector_db_path,
            "config": config_path
        }
        
    @classmethod
    def load(cls, directory: str = "data/rag_pipeline") -> "RAGPipeline":
        """
        Load a RAG pipeline from saved components.
        
        Args:
            directory: Directory containing saved components
            
        Returns:
            Loaded RAG pipeline
        """
        # Load configuration
        config_path = os.path.join(directory, "rag_config.json")
        
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Load vector database
        vector_db = VectorDatabase.load(
            prefix=config["vector_db_prefix"],
            storage_dir=directory
        )
        
        # Create document processor
        document_processor = DocumentProcessor(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        
        # Create embedding generator
        embedding_generator = EmbeddingGenerator(
            model_name=config["embedding_model"]
        )
        
        # Create RAG pipeline
        pipeline = cls(
            document_processor=document_processor,
            embedding_generator=embedding_generator,
            vector_database=vector_db,
            top_k=config["top_k"]
        )
        
        pipeline.logger.info(f"Loaded RAG pipeline from {directory}")
        
        return pipeline
