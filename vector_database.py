"""
Vector database interface for RAG pipeline.
Handles storage and retrieval of document embeddings.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import faiss
import pickle
from dataclasses import asdict

from src.rag.document_processor import Document


class VectorDatabase:
    """
    Vector database for storing and retrieving document embeddings.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        dimension: int = 384,  # Default for all-MiniLM-L6-v2
        index_type: str = "flat",  # "flat", "ivf", "hnsw"
        metric_type: str = "cosine",  # "cosine", "l2", "ip" (inner product)
        storage_dir: str = "data/embeddings",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the vector database.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use
            metric_type: Distance metric for similarity search
            storage_dir: Directory to store the index and metadata
            logger: Logger instance
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.storage_dir = storage_dir
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Document storage (id -> document)
        self.documents = {}
        
        # Mapping from index position to document ID
        self.index_to_id = []
        
        self.logger.info(f"Initialized vector database with {index_type} index and {metric_type} metric")
        
    def _create_index(self) -> faiss.Index:
        """
        Create a FAISS index based on the specified parameters.
        
        Returns:
            FAISS index
        """
        # Set up the metric type
        if self.metric_type == "cosine":
            # For cosine similarity, we use inner product on normalized vectors
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric_type == "l2":
            metric = faiss.METRIC_L2
        elif self.metric_type == "ip":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
            
        # Create the appropriate index
        if self.index_type == "flat":
            # Flat index (exact search)
            index = faiss.IndexFlatIP(self.dimension) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "ivf":
            # IVF index (approximate search with inverted file)
            # We need a quantizer first
            quantizer = faiss.IndexFlatIP(self.dimension) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)
            # Number of centroids - rule of thumb: sqrt(n) where n is expected dataset size
            # Here we use 100 as a reasonable default
            nlist = 100
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
            # IVF indices need to be trained before use
            # We'll train it when data is added
            
        elif self.index_type == "hnsw":
            # HNSW index (hierarchical navigable small world graph)
            index = faiss.IndexHNSWFlat(self.dimension, 32, metric)  # 32 is M (number of connections per node)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        return index
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents with embeddings to the database.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        # Extract embeddings and IDs
        embeddings = []
        doc_ids = []
        
        for doc in documents:
            # Ensure document has an embedding
            if "embedding" not in doc:
                self.logger.warning(f"Document {doc.get('id', 'unknown')} has no embedding, skipping")
                continue
                
            # Convert embedding to numpy array if it's a list
            embedding = np.array(doc["embedding"], dtype=np.float32)
            
            # Ensure embedding has the correct dimension
            if embedding.shape[0] != self.dimension:
                self.logger.warning(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {embedding.shape[0]}, skipping document {doc.get('id', 'unknown')}"
                )
                continue
                
            # Normalize embedding for cosine similarity if needed
            if self.metric_type == "cosine":
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
            embeddings.append(embedding)
            doc_ids.append(doc["id"])
            
            # Store document metadata
            self.documents[doc["id"]] = {
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"]
            }
            
            # Update index mapping
            self.index_to_id.append(doc["id"])
            
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings_array)
            
        # Add vectors to index
        self.index.add(embeddings_array)
        
        self.logger.info(f"Added {len(doc_ids)} documents to vector database")
        
        return doc_ids
        
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        # Ensure query embedding has the correct dimension
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[0]}"
            )
            
        # Normalize query embedding for cosine similarity if needed
        if self.metric_type == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
                
        # Reshape query embedding to 2D array
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search index
        # We request more results than needed to account for filtering
        num_to_request = min(k * 4, len(self.index_to_id)) if filter_dict else k
        num_to_request = max(num_to_request, k)  # Ensure we request at least k
        
        if num_to_request == 0:
            return []
            
        distances, indices = self.index.search(query_embedding, num_to_request)
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.index_to_id):
                continue
                
            # Get document ID
            doc_id = self.index_to_id[idx]
            
            # Get document
            doc = self.documents.get(doc_id)
            if not doc:
                continue
                
            # Apply filter if provided
            if filter_dict and not self._matches_filter(doc, filter_dict):
                continue
                
            # Convert distance to score (for cosine and inner product)
            score = distance
            if self.metric_type == "l2":
                # Convert L2 distance to similarity score (smaller is better)
                score = 1.0 / (1.0 + distance)
                
            # Add to results
            results.append({
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score)
            })
            
            # Stop if we have enough results
            if len(results) >= k:
                break
                
        return results
        
    def _matches_filter(self, doc: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if document matches the filter.
        
        Args:
            doc: Document to check
            filter_dict: Filter dictionary
            
        Returns:
            True if document matches filter, False otherwise
        """
        metadata = doc.get("metadata", {})
        
        for key, value in filter_dict.items():
            # Skip if key doesn't exist in metadata
            if key not in metadata:
                return False
                
            # Check if value matches
            if metadata[key] != value:
                return False
                
        return True
        
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the database.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        # Check if document exists
        if doc_id not in self.documents:
            return False
            
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        # This is inefficient for frequent deletions, but works for occasional ones
        
        # Remove document from storage
        del self.documents[doc_id]
        
        # Find indices to keep
        keep_indices = []
        new_index_to_id = []
        
        for i, idx_doc_id in enumerate(self.index_to_id):
            if idx_doc_id != doc_id:
                keep_indices.append(i)
                new_index_to_id.append(idx_doc_id)
                
        # Rebuild index if there are documents left
        if keep_indices:
            # Create a new index
            new_index = self._create_index()
            
            # Extract vectors to keep
            keep_indices = np.array(keep_indices)
            vectors_to_keep = faiss.vector_to_array(self.index.reconstruct_n(0, len(self.index_to_id)))
            vectors_to_keep = vectors_to_keep.reshape(len(self.index_to_id), self.dimension)
            vectors_to_keep = vectors_to_keep[keep_indices]
            
            # Train new index if needed
            if self.index_type == "ivf" and not new_index.is_trained:
                new_index.train(vectors_to_keep)
                
            # Add vectors to new index
            new_index.add(vectors_to_keep)
            
            # Replace old index
            self.index = new_index
            self.index_to_id = new_index_to_id
        else:
            # No documents left, reset index
            self.index = self._create_index()
            self.index_to_id = []
            
        self.logger.info(f"Deleted document {doc_id} from vector database")
        
        return True
        
    def save(self, prefix: Optional[str] = None) -> str:
        """
        Save the vector database to disk.
        
        Args:
            prefix: Optional prefix for the saved files
            
        Returns:
            Path to the saved index
        """
        # Create prefix if not provided
        if prefix is None:
            prefix = f"index_{self.index_type}_{self.metric_type}_{self.dimension}"
            
        # Create full paths
        index_path = os.path.join(self.storage_dir, f"{prefix}.index")
        metadata_path = os.path.join(self.storage_dir, f"{prefix}.metadata")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "documents": self.documents,
            "index_to_id": self.index_to_id
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
            
        self.logger.info(f"Saved vector database to {index_path} and {metadata_path}")
        
        return index_path
        
    @classmethod
    def load(cls, prefix: str, storage_dir: str = "data/embeddings") -> "VectorDatabase":
        """
        Load a vector database from disk.
        
        Args:
            prefix: Prefix of the saved files
            storage_dir: Directory where the files are stored
            
        Returns:
            Loaded vector database
        """
        # Create full paths
        index_path = os.path.join(storage_dir, f"{prefix}.index")
        metadata_path = os.path.join(storage_dir, f"{prefix}.metadata")
        
        # Check if files exist
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index or metadata file not found: {index_path}, {metadata_path}")
            
        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            
        # Create instance
        instance = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric_type=metadata["metric_type"],
            storage_dir=storage_dir
        )
        
        # Load FAISS index
        instance.index = faiss.read_index(index_path)
        
        # Load metadata
        instance.documents = metadata["documents"]
        instance.index_to_id = metadata["index_to_id"]
        
        instance.logger.info(f"Loaded vector database from {index_path} and {metadata_path}")
        
        return instance
