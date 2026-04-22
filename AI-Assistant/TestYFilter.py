"""
Unit Tests — Academic Year Filter
Tests for document_processor.py, retriever.py, and rag_pipeline.py

Run with:
    pytest test_year_filter.py -v

Or without pytest:
    python test_year_filter.py
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


# ─────────────────────────────────────────────
# SECTION 1: DocumentProcessor Tests
# ─────────────────────────────────────────────

class TestExtractYearFromPath(unittest.TestCase):
    """Test _extract_year_from_path in DocumentProcessor."""

    def _get_processor(self):
        """Create a DocumentProcessor instance with all dependencies mocked."""
        with patch("src.document_processor.RecursiveCharacterTextSplitter"), \
             patch("src.document_processor.CHUNK_SIZE", 500), \
             patch("src.document_processor.CHUNK_OVERLAP", 50), \
             patch("src.document_processor.ENABLE_OCR", False), \
             patch("src.document_processor.TESSERACT_CMD", ""):
            from src.document_processor import DocumentProcessor
            return DocumentProcessor()

    def test_year_4_from_path(self):
        """Should extract '4' from a path containing 'year_4'."""
        processor = self._get_processor()
        path = Path("/data/year_4/math/lecture1.pdf")
        result = processor._extract_year_from_path(path)
        self.assertEqual(result, "4")

    def test_year_1_from_path(self):
        """Should extract '1' from a path containing 'year_1'."""
        processor = self._get_processor()
        path = Path("/data/year_1/physics/slides.pptx")
        result = processor._extract_year_from_path(path)
        self.assertEqual(result, "1")

    def test_year_without_underscore(self):
        """Should handle 'year2' without underscore."""
        processor = self._get_processor()
        path = Path("/data/year2/chemistry/notes.pdf")
        result = processor._extract_year_from_path(path)
        self.assertEqual(result, "2")

    def test_no_year_in_path(self):
        """Should return 'unknown' when no year folder found."""
        processor = self._get_processor()
        path = Path("/data/general/lecture.pdf")
        result = processor._extract_year_from_path(path)
        self.assertEqual(result, "unknown")

    def test_year_uppercase(self):
        """Should handle 'Year_3' with uppercase."""
        processor = self._get_processor()
        path = Path("/uploads/Year_3/biology/doc.pdf")
        result = processor._extract_year_from_path(path)
        self.assertEqual(result, "3")

    def test_non_digit_after_year(self):
        """Should return 'unknown' if folder is 'year_abc' (non-digit)."""
        processor = self._get_processor()
        path = Path("/data/year_abc/lecture.pdf")
        result = processor._extract_year_from_path(path)
        self.assertEqual(result, "unknown")


class TestEnrichMetadata(unittest.TestCase):
    """Test that _enrich_metadata correctly sets academic_year."""

    def _get_processor(self):
        with patch("src.document_processor.RecursiveCharacterTextSplitter"), \
             patch("src.document_processor.CHUNK_SIZE", 500), \
             patch("src.document_processor.CHUNK_OVERLAP", 50), \
             patch("src.document_processor.ENABLE_OCR", False), \
             patch("src.document_processor.TESSERACT_CMD", ""):
            from src.document_processor import DocumentProcessor
            return DocumentProcessor()

    def _make_doc(self):
        from langchain_core.documents import Document
        return Document(page_content="Sample content", metadata={})

    def test_academic_year_in_metadata(self):
        """academic_year should appear in document metadata after enrichment."""
        processor = self._get_processor()
        doc = self._make_doc()
        path = Path("/data/year_3/subject/file.pdf")
        processor._enrich_metadata([doc], path)
        self.assertIn("academic_year", doc.metadata)
        self.assertEqual(doc.metadata["academic_year"], "3")

    def test_standard_metadata_still_present(self):
        """Other metadata fields should still be set alongside academic_year."""
        processor = self._get_processor()
        doc = self._make_doc()
        path = Path("/data/year_2/file.pdf")
        processor._enrich_metadata([doc], path)
        self.assertIn("file_name", doc.metadata)
        self.assertIn("file_type", doc.metadata)
        self.assertIn("source", doc.metadata)
        self.assertIn("ingestion_timestamp", doc.metadata)

    def test_unknown_year_when_no_folder(self):
        """academic_year should be 'unknown' if path has no year folder."""
        processor = self._get_processor()
        doc = self._make_doc()
        path = Path("/data/shared/file.pdf")
        processor._enrich_metadata([doc], path)
        self.assertEqual(doc.metadata["academic_year"], "unknown")

    def test_multiple_documents_get_same_year(self):
        """All documents from same path should get the same academic_year."""
        processor = self._get_processor()
        docs = [self._make_doc(), self._make_doc(), self._make_doc()]
        path = Path("/data/year_4/subject/file.pdf")
        processor._enrich_metadata(docs, path)
        for doc in docs:
            self.assertEqual(doc.metadata["academic_year"], "4")


# ─────────────────────────────────────────────
# SECTION 2: Retriever Tests
# ─────────────────────────────────────────────

class TestRetrieverYearFilter(unittest.TestCase):
    """Test that Retriever passes academic_year filter to ChromaDB."""

    def _get_retriever(self):
        with patch("src.retriever.VectorStoreManager"), \
             patch("src.retriever.TOP_K_RESULTS", 5), \
             patch("src.retriever.SIMILARITY_THRESHOLD", 0):
            from src.retriever import Retriever
            return Retriever()

    def _setup_mock_store(self, retriever, docs_with_scores):
        """Helper: attach a mock vector store that returns given results."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = docs_with_scores
        retriever.vector_store_manager.get_vector_store.return_value = mock_store
        return mock_store

    def _make_doc(self, content="Test content"):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata={"academic_year": "4"})

    def test_filter_passed_when_year_given(self):
        """similarity_search_with_score should be called with correct year filter."""
        retriever = self._get_retriever()
        mock_store = self._setup_mock_store(retriever, [(self._make_doc(), 0.1)])

        retriever.retrieve("What is an algorithm?", academic_year="4")

        call_kwargs = mock_store.similarity_search_with_score.call_args
        passed_filter = call_kwargs[1].get("filter") or call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs[1].get("filter")
        self.assertEqual(passed_filter, {"academic_year": {"$eq": "4"}})

    def test_no_filter_when_year_is_none(self):
        """filter should be None when no academic_year is provided."""
        retriever = self._get_retriever()
        mock_store = self._setup_mock_store(retriever, [(self._make_doc(), 0.1)])

        retriever.retrieve("What is an algorithm?", academic_year=None)

        call_kwargs = mock_store.similarity_search_with_score.call_args
        passed_filter = call_kwargs[1].get("filter")
        self.assertIsNone(passed_filter)

    def test_no_filter_when_year_is_unknown(self):
        """filter should be None when academic_year is 'unknown'."""
        retriever = self._get_retriever()
        mock_store = self._setup_mock_store(retriever, [(self._make_doc(), 0.1)])

        retriever.retrieve("Some question", academic_year="unknown")

        call_kwargs = mock_store.similarity_search_with_score.call_args
        passed_filter = call_kwargs[1].get("filter")
        self.assertIsNone(passed_filter)

    def test_returns_documents(self):
        """retrieve() should return Document objects."""
        retriever = self._get_retriever()
        doc = self._make_doc("Data structures lecture content")
        self._setup_mock_store(retriever, [(doc, 0.2)])

        results = retriever.retrieve("data structures", academic_year="2")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Data structures lecture content")

    def test_returns_empty_when_no_results(self):
        """retrieve() should return empty list when store has no matches."""
        retriever = self._get_retriever()
        self._setup_mock_store(retriever, [])

        results = retriever.retrieve("quantum physics", academic_year="1")

        self.assertEqual(results, [])


# ─────────────────────────────────────────────
# SECTION 3: RAGPipeline Tests
# ─────────────────────────────────────────────

class TestRAGPipelineYearPropagation(unittest.TestCase):
    """Test that RAGPipeline.query() passes academic_year down to retriever."""

    def _get_pipeline(self):
        with patch("src.rag_pipeline.DocumentProcessor"), \
             patch("src.rag_pipeline.VectorStoreManager"), \
             patch("src.rag_pipeline.Retriever"), \
             patch("src.rag_pipeline.Generator"), \
             patch("src.rag_pipeline.YouTubeProcessor"), \
             patch("src.rag_pipeline.RecommendationEngine"), \
             patch("src.rag_pipeline.PresentationMaker"), \
             patch("src.rag_pipeline.RAW_DATA_DIR", "/data"):
            from src.rag_pipeline import RAGPipeline
            pipeline = RAGPipeline()
            pipeline.is_initialized = True
            return pipeline

    def _make_doc(self, content="Lecture content about algorithms"):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata={"source": "year_4/algo/lec1.pdf"})

    def test_academic_year_passed_to_retriever(self):
        """query() should call retriever.retrieve() with correct academic_year."""
        pipeline = self._get_pipeline()

        # Mock retriever to return a doc
        pipeline.retriever.retrieve.return_value = [self._make_doc()]

        # Mock generator
        pipeline.generator.generate_answer.return_value = "Here is your answer."

        # Mock youtube processor (no URL)
        pipeline.youtube_processor.process_url.return_value = None

        pipeline.query("What is Big O notation?", academic_year="4")

        # Check retriever was called with academic_year="4"
        call_kwargs = pipeline.retriever.retrieve.call_args
        passed_year = call_kwargs[1].get("academic_year") or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        self.assertEqual(passed_year, "4")

    def test_academic_year_none_by_default(self):
        """query() without academic_year should pass None to retriever."""
        pipeline = self._get_pipeline()
        pipeline.retriever.retrieve.return_value = [self._make_doc()]
        pipeline.generator.generate_answer.return_value = "Answer."
        pipeline.youtube_processor.process_url.return_value = None

        pipeline.query("What is recursion?")

        call_kwargs = pipeline.retriever.retrieve.call_args
        passed_year = call_kwargs[1].get("academic_year")
        self.assertIsNone(passed_year)

    def test_query_returns_answer(self):
        """query() should return a dict with 'answer' key."""
        pipeline = self._get_pipeline()
        pipeline.retriever.retrieve.return_value = [self._make_doc()]
        pipeline.generator.generate_answer.return_value = "Big O describes time complexity."
        pipeline.youtube_processor.process_url.return_value = None

        result = pipeline.query("Explain Big O", academic_year="3")

        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "Big O describes time complexity.")

    def test_different_years_isolated(self):
        """Calling query() with year=1 and year=4 should pass each year correctly."""
        pipeline = self._get_pipeline()
        pipeline.retriever.retrieve.return_value = [self._make_doc()]
        pipeline.generator.generate_answer.return_value = "Answer."
        pipeline.youtube_processor.process_url.return_value = None

        pipeline.query("Question", academic_year="1")
        first_call_year = pipeline.retriever.retrieve.call_args[1].get("academic_year")

        pipeline.query("Question", academic_year="4")
        second_call_year = pipeline.retriever.retrieve.call_args[1].get("academic_year")

        self.assertEqual(first_call_year, "1")
        self.assertEqual(second_call_year, "4")


# ─────────────────────────────────────────────
# Runner (بدون pytest)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Academic Year Filter — Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)