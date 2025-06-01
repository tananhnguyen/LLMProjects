"""
Medical RAG pipeline for PubMed data integration.
Extends the base RAG pipeline with medical domain-specific functionality.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from src.rag.document_processor import Document, DocumentProcessor
from src.rag.embedding_generator import EmbeddingGenerator
from src.rag.vector_database import VectorDatabase
from src.rag.rag_pipeline import RAGPipeline
from src.pubmed.pubmed_client import PubMedClient, PubMedConfig


class MedicalDocumentProcessor(DocumentProcessor):
    """
    Document processor specialized for medical text.
    Extends the base DocumentProcessor with medical-specific preprocessing.
    """
    
    def __init__(
        self,
        chunk_size: int = 800,  # Smaller chunks for medical text
        chunk_overlap: int = 200,
        metadata_fields: List[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the medical document processor.
        
        Args:
            chunk_size: Maximum size of each document chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            metadata_fields: List of metadata fields to extract
            logger: Logger instance
        """
        super().__init__(chunk_size, chunk_overlap, metadata_fields, logger)
        
        # Default metadata fields for medical documents
        if metadata_fields is None:
            self.metadata_fields = [
                "pmid", "doi", "authors", "journal", "pubdate", "source"
            ]
            
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess medical text content.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Apply base preprocessing
        text = super().preprocess_text(text)
        
        # Medical-specific preprocessing
        
        # Replace common medical abbreviations
        medical_abbr = {
            "pt": "patient",
            "pts": "patients",
            "dx": "diagnosis",
            "tx": "treatment",
            "hx": "history",
            "fx": "fracture",
            "sx": "symptoms",
            "RCT": "randomized controlled trial",
            "w/": "with",
            "w/o": "without",
            "yo": "year old",
            "y/o": "year old"
        }
        
        for abbr, full in medical_abbr.items():
            # Only replace if it's a standalone word
            text = text.replace(f" {abbr} ", f" {full} ")
            
        # Handle common medical units
        text = text.replace(" mg/kg ", " milligrams per kilogram ")
        text = text.replace(" mcg ", " micrograms ")
        
        return text
        
    def process_pubmed_article(self, article: Dict[str, Any]) -> List[Document]:
        """
        Process a PubMed article into document chunks.
        
        Args:
            article: PubMed article data
            
        Returns:
            List of document chunks
        """
        # Extract content and metadata
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        
        # Combine title and abstract
        content = f"{title}\n\n{abstract}"
        
        # Extract metadata
        metadata = {
            "pmid": article.get("pmid", ""),
            "doi": article.get("doi", ""),
            "authors": article.get("authors", ""),
            "journal": article.get("journal", ""),
            "pubdate": article.get("pubdate", ""),
            "title": title,
            "source": "PubMed"
        }
        
        # Create document
        doc = Document(content=content, metadata=metadata)
        
        # Process document
        chunks = self.process_document(doc)
        
        return chunks


class MedicalEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator specialized for medical text.
    Uses biomedical-specific embedding models.
    """
    
    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",  # Medical-specific model
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the medical embedding generator.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run the model on
            max_length: Maximum sequence length
            batch_size: Batch size for embedding generation
            normalize_embeddings: Whether to normalize embeddings
            cache_dir: Directory to cache the model
            logger: Logger instance
        """
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            cache_dir=cache_dir,
            logger=logger
        )


class MedicalRAGPipeline:
    """
    Medical-specific RAG pipeline for PubMed data.
    """
    
    def __init__(
        self,
        document_processor: Optional[MedicalDocumentProcessor] = None,
        embedding_generator: Optional[MedicalEmbeddingGenerator] = None,
        vector_database: Optional[VectorDatabase] = None,
        pubmed_client: Optional[PubMedClient] = None,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the medical RAG pipeline.
        
        Args:
            document_processor: Medical document processor instance
            embedding_generator: Medical embedding generator instance
            vector_database: Vector database instance
            pubmed_client: PubMed client instance
            top_k: Number of documents to retrieve
            logger: Logger instance
        """
        # Initialize components with defaults if not provided
        self.document_processor = document_processor or MedicalDocumentProcessor()
        self.embedding_generator = embedding_generator or MedicalEmbeddingGenerator()
        self.vector_database = vector_database or VectorDatabase(
            dimension=768  # Default for PubMedBERT
        )
        self.pubmed_client = pubmed_client or PubMedClient()
        
        # Create base RAG pipeline
        self.rag_pipeline = RAGPipeline(
            document_processor=self.document_processor,
            embedding_generator=self.embedding_generator,
            vector_database=self.vector_database,
            top_k=top_k
        )
        
        self.top_k = top_k
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Initialized medical RAG pipeline")
        
    def add_pubmed_articles(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add PubMed articles to the RAG pipeline.
        
        Args:
            articles: List of PubMed articles
            
        Returns:
            List of document IDs
        """
        self.logger.info(f"Adding {len(articles)} PubMed articles to RAG pipeline")
        
        # Process articles
        all_chunks = []
        for article in articles:
            chunks = self.document_processor.process_pubmed_article(article)
            all_chunks.extend(chunks)
            
        # Generate embeddings
        embedded_docs = self.embedding_generator.embed_documents(all_chunks)
        
        # Add to vector database
        doc_ids = self.vector_database.add_documents(embedded_docs)
        
        self.logger.info(f"Added {len(doc_ids)} document chunks from {len(articles)} articles")
        
        return doc_ids
        
    def search_and_add_pubmed(
        self,
        query: str,
        max_results: int = 100
    ) -> List[str]:
        """
        Search PubMed for articles and add them to the RAG pipeline.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of document IDs
        """
        self.logger.info(f"Searching PubMed for: {query}")
        
        # Search PubMed
        articles = self.pubmed_client.search_and_fetch(query, max_results)
        
        # Add articles to RAG pipeline
        doc_ids = self.add_pubmed_articles(articles)
        
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
            top_k: Number of documents to retrieve
            filter_dict: Optional metadata filter
            
        Returns:
            List of retrieved documents with scores
        """
        # Use the base RAG pipeline's retrieve method
        return self.rag_pipeline.retrieve(query, top_k, filter_dict)
        
    def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        format_template: str = "Medical Context:\n{context}\n\nMedical Question: {query}\n\nAnswer:"
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
        # Use the base RAG pipeline's retrieve method
        results = self.retrieve(query, top_k, filter_dict)
        
        # Extract content from results
        contexts = []
        for result in results:
            content = result["content"]
            metadata = result["metadata"]
            
            # Add source information
            source_info = f"[Source: {metadata.get('journal', 'Unknown')}"
            if metadata.get('pubdate'):
                source_info += f", {metadata.get('pubdate')}"
            if metadata.get('authors'):
                source_info += f", {metadata.get('authors')}"
            source_info += "]"
            
            contexts.append(f"{content}\n{source_info}")
            
        # Join contexts
        context_text = "\n\n".join(contexts)
        
        # Format prompt
        formatted_prompt = format_template.format(
            context=context_text,
            query=query
        )
        
        return formatted_prompt, results
        
    def save(self, directory: str = "data/medical_rag") -> Dict[str, str]:
        """
        Save the medical RAG pipeline components.
        
        Args:
            directory: Directory to save components
            
        Returns:
            Dictionary of saved file paths
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save vector database
        vector_db_path = self.vector_database.save(prefix="medical_rag")
        
        # Save configuration
        config = {
            "top_k": self.top_k,
            "vector_db_prefix": "medical_rag",
            "embedding_model": self.embedding_generator.model_name,
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap
        }
        
        import json
        config_path = os.path.join(directory, "medical_rag_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Saved medical RAG pipeline to {directory}")
        
        return {
            "vector_db": vector_db_path,
            "config": config_path
        }
        
    @classmethod
    def load(cls, directory: str = "data/medical_rag") -> "MedicalRAGPipeline":
        """
        Load a medical RAG pipeline from saved components.
        
        Args:
            directory: Directory containing saved components
            
        Returns:
            Loaded medical RAG pipeline
        """
        # Load configuration
        config_path = os.path.join(directory, "medical_rag_config.json")
        
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Load vector database
        vector_db = VectorDatabase.load(
            prefix=config["vector_db_prefix"],
            storage_dir=directory
        )
        
        # Create document processor
        document_processor = MedicalDocumentProcessor(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        
        # Create embedding generator
        embedding_generator = MedicalEmbeddingGenerator(
            model_name=config["embedding_model"]
        )
        
        # Create medical RAG pipeline
        pipeline = cls(
            document_processor=document_processor,
            embedding_generator=embedding_generator,
            vector_database=vector_db,
            top_k=config["top_k"]
        )
        
        pipeline.logger.info(f"Loaded medical RAG pipeline from {directory}")
        
        return pipeline


# Import necessary modules
import torch
