"""
Module for chunking complaint narratives.
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextChunker:
    """Handles chunking of complaint narratives."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size (int): Maximum size of each text chunk.
            chunk_overlap (int): Overlap between chunks for context continuity.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_narratives(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
        """
        Chunk complaint narratives and store metadata.
        
        Args:
            df (pd.DataFrame): DataFrame with complaint data.
        
        Returns:
            Tuple[List[str], List[Dict]]: List of chunks and corresponding metadata.
        """
        try:
            chunks = []
            metadata = []
            for _, row in df.iterrows():
                narrative = row['consumer_complaint_narrative']
                if not isinstance(narrative, str) or not narrative.strip():
                    continue
                split_texts = self.text_splitter.split_text(narrative)
                for i, chunk in enumerate(split_texts):
                    chunks.append(chunk)
                    metadata.append({
                        'complaint_id': str(row['complaint_id']),
                        'product': row['product'],
                        'chunk_index': i,
                        'original_text_length': len(narrative)
                    })
            logger.info(f"Created {len(chunks)} chunks from {len(df)} narratives")
            return chunks, metadata
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            raise