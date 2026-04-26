"""
One-time script to re-ingest all documents with Course Code metadata.
Deletes the existing vector store and rebuilds it.
Run: python reingest_with_codes.py
"""
import sys, os
from pathlib import Path

# Add the backend and ai_engine directories to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'ai_engine'))

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ai_engine.src.rag_pipeline import RAGPipeline
    from ai_engine.config.settings import PROCESSED_DATA_DIR
    import shutil

    # 1. Clear existing processed data
    logger.info("Cleaning up old vector store and cache...")
    chroma_dir = PROCESSED_DATA_DIR / "chroma_db"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        logger.info(f"Deleted {chroma_dir}")
        
    cache_file = PROCESSED_DATA_DIR / "processed_chunks.jsonl"
    if cache_file.exists():
        os.remove(cache_file)
        logger.info(f"Deleted {cache_file}")

    # 2. Initialize Pipeline (Re-ingest)
    logger.info("Starting re-ingestion with Course Codes...")
    pipeline = RAGPipeline()
    pipeline.initialize()
    
    logger.info("Re-ingestion COMPLETE! All chunks are now tagged with course_code and doc_category.")

except Exception as e:
    logger.error(f"Re-ingestion failed: {e}")
    import traceback
    traceback.print_exc()
