from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.embeddings import EmbeddingManager
from config.settings import CHROMA_DB_DIR, COLLECTION_NAME , _ADD_BATCH_SIZE

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



class VectorStoreError(Exception):
    """Base exception for VectorStoreManager failures."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when load_vector_store is called but no persisted store exists."""


class VectorStoreManager:
    
    def __init__(
        self,
        persist_directory: str = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()

        self._vector_store: Optional[Chroma] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_vector_store(
        self,
        documents: List[Document],
        overwrite: bool = False,
    ) -> Chroma:
        
        if not documents:
            raise ValueError("Cannot create a vector store from an empty document list.")

        if os.path.exists(self.persist_directory):
            if not overwrite:
                raise VectorStoreError(
                    f"A vector store already exists at '{self.persist_directory}'. "
                    "Pass overwrite=True to replace it."
                )
            self._remove_persist_directory()

        logger.info(
            f"Creating vector store with {len(documents)} document(s) "
            f"at '{self.persist_directory}' …"
        )
        embeddings = self.embedding_manager.get_embeddings()

        with self._lock:
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

        count = self._safe_count()
        logger.info(
            f"Vector store created: {count} document(s) persisted "
            f"to '{self.persist_directory}'."
        )
        return self._vector_store

    def load_vector_store(self) -> Chroma:
        
        if not os.path.exists(self.persist_directory) or not os.listdir(
            self.persist_directory
        ):
            raise VectorStoreNotFoundError(
                f"No vector store found at '{self.persist_directory}'. "
                "Run create_vector_store() first."
            )

        logger.info(f"Loading vector store from '{self.persist_directory}' …")
        embeddings = self.embedding_manager.get_embeddings()

        with self._lock:
            self._vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )

        count = self._safe_count()
        logger.info(f"Vector store loaded: {count} document(s) in collection.")
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        
        if not documents:
            logger.warning("add_documents called with an empty list; nothing to do.")
            return

        store = self.get_vector_store()
        total = len(documents)
        logger.info(f"Adding {total} document(s) in batches of {_ADD_BATCH_SIZE} …")

        for start in range(0, total, _ADD_BATCH_SIZE):
            batch = documents[start : start + _ADD_BATCH_SIZE]
            try:
                store.add_documents(batch)
                logger.debug(
                    f"Batch {start // _ADD_BATCH_SIZE + 1}: "
                    f"added documents {start + 1}–{start + len(batch)} of {total}."
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to add batch starting at index {start}: {exc}"
                ) from exc

        logger.info(f"Successfully added {total} document(s) to the vector store.")

    def get_vector_store(self) -> Chroma:
        
        if self._vector_store is not None:
            return self._vector_store

        with self._lock:
            # Double-checked locking: another thread may have loaded the
            # store while we were waiting.
            if self._vector_store is None:
                self.load_vector_store()

        return self._vector_store  # type: ignore[return-value]

    def store_exists(self) -> bool:
        """Return ``True`` if a persisted vector store exists on disk."""
        return bool(
            os.path.exists(self.persist_directory)
            and os.listdir(self.persist_directory)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_persist_directory(self) -> None:
        """Delete the persist directory, raising VectorStoreError on failure."""
        try:
            shutil.rmtree(self.persist_directory)
            self._vector_store = None
            logger.info(f"Removed existing vector store at '{self.persist_directory}'.")
        except OSError as exc:
            raise VectorStoreError(
                f"Could not remove existing vector store at "
                f"'{self.persist_directory}': {exc}"
            ) from exc

    def _safe_count(self) -> int:
        
        try:
            return self._vector_store._collection.count()  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug(f"Could not retrieve document count: {exc}")
            return -1