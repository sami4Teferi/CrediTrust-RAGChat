"""
Module for generating text embeddings.
"""

import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

class Embedder:
    """Handles generation of text embeddings."""
    
    def __init__(self, model_name: str):
        """
        Initialize the Embedder.
        
        Args:
            model_name (str): Name of the embedding model.
        """
        self.model_name = model_name
        self.model = None

    def initialize_model(self):
        """Load the embedding model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts (List[str]): List of text chunks to embed.
            batch_size (int): Batch size for embedding.
        
        Returns:
            Optional[np.ndarray]: Array of embeddings or None if generation fails.
        """
        try:
            if not self.model:
                self.initialize_model()
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Generated embeddings for {len(texts)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None