"""
Main script to orchestrate Task 2: Text Chunking, Embedding, and Vector Store Indexing.
"""

import logging
from scripts.config import Config
from scripts.data_loader import DataLoader
from scripts.text_chunker import TextChunker
from scripts.embedder import Embedder
from scripts.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Execute the Task 2 pipeline."""
    try:
        logger.info("Starting Task 2: Chunking, Embedding, and Indexing")
        
        # Initialize components
        config = Config()
        loader = DataLoader(config.DATA_PATH)
        chunker = TextChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        embedder = Embedder(config.EMBEDDING_MODEL_NAME)
        vector_store = VectorStore(config.VECTOR_STORE_DIR)

        # Load data
        df = loader.load_data()
        if df is None:
            raise ValueError("Failed to load data")

        # Chunk narratives
        chunks, metadata = chunker.chunk_narratives(df)
        if not chunks:
            raise ValueError("No chunks generated")

        # Generate embeddings
        embeddings = embedder.generate_embeddings(chunks, batch_size=config.BATCH_SIZE)
        if embeddings is None:
            raise ValueError("Failed to generate embeddings")

        # Create and save vector store
        vector_store.create_vector_store(embeddings)
        vector_store.save_vector_store(metadata)

        logger.info("Task 2 completed successfully")
    except Exception as e:
        logger.error(f"Task 2 failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()