"""
Configuration settings for Task 2: Text Chunking, Embedding, and Vector Store Indexing.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration class for Task 2 parameters."""
    DATA_PATH: Path = Path("data/filtered_complaints.csv")
    VECTOR_STORE_DIR: Path = Path("vector_store")
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE: int = 32
    LOG_FILE: Path = Path("logs/task2.log")