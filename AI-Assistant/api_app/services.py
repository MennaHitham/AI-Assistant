import os
import logging
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Directory for presentation images
IMAGE_DIR = os.path.join("data", "presentation_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

class RAGService:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RAGPipeline()
            # Try to load existing vector store
            try:
                cls._instance.vector_store_manager.load_vector_store()
                count = cls._instance.vector_store_manager.vector_store._collection.count()
                if count > 0:
                    cls._instance.is_initialized = True
                    logger.info(f"Vector store loaded on startup ({count} documents)")
                else:
                    logger.info("Vector store is empty — call /initialize to build it")
            except Exception as e:
                logger.info(f"No existing vector store found: {e}")
        return cls._instance

rag = RAGService.get_instance()
