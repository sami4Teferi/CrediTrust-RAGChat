"""
Module for creating and saving a FAISS vector store.
"""

import logging
import faiss
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles creation and storage of a FAISS vector store."""
    
    def __init__(self, vector_store_dir: Path):
        """
        Initialize the VectorStore.
        
        Args:
            vector_store_dir (Path): Directory to save the vector store.
        """
        self.vector_store_dir = vector_store_dir
        self.index = None

    def create_vector_store(self, embeddings: np.ndarray):
        """
        Create a FAISS vector store from embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings.
        """
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def save_vector_store(self, metadata: List[dict]):
        """
        Save the vector store and metadata to disk.
        
        Args:
            metadata (List[dict]): List of metadata for each embedding.
        """
        try:
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            faiss.write_index(self.index, str(self.vector_store_dir / f"faiss_index_{timestamp}.index"))
            with open(self.vector_store_dir / f"metadata_{timestamp}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved vector store and metadata to {self.vector_store_dir}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise