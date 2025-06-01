"""
Embedding generator module for RAG pipeline.
Handles text embedding generation for vector search.
"""

import torch
from typing import Dict, List, Optional, Union, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging

from src.rag.document_processor import Document


class EmbeddingGenerator:
    """
    Generator for text embeddings used in vector search.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run the model on
            max_length: Maximum sequence length
            batch_size: Batch size for embedding generation
            normalize_embeddings: Whether to normalize embeddings
            cache_dir: Directory to cache the model
            logger: Logger instance
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Loading embedding model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _mean_pooling(
        self, 
        model_output: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        
        Args:
            model_output: Model output containing token embeddings
            attention_mask: Attention mask for the input
            
        Returns:
            Pooled embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        # Expand attention mask from [batch_size, seq_length] to [batch_size, seq_length, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum token embeddings and divide by the expanded mask sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform pooling
        embedding = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        
        # Normalize embedding
        if self.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        # Convert to numpy array
        embedding_np = embedding.cpu().numpy()[0]
        
        return embedding_np
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize batch
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_inputs)
                
            # Perform pooling
            embeddings = self._mean_pooling(
                model_output, encoded_inputs["attention_mask"]
            )
            
            # Normalize embeddings
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            # Convert to numpy and add to results
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
            
        # Concatenate all batches
        return np.vstack(all_embeddings)
        
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of documents with embeddings
        """
        # Extract text content from documents
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to documents
        embedded_docs = []
        for i, doc in enumerate(documents):
            doc_dict = {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": embeddings[i].tolist()
            }
            embedded_docs.append(doc_dict)
            
        self.logger.info(f"Generated embeddings for {len(documents)} documents")
        
        return embedded_docs
