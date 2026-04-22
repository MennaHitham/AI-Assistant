"""
test_chatbot.py
===============
Tests for the RAG chatbot — Year 2 / Stack questions.

Run unit tests only (no real pipeline):
    pytest test_chatbot.py -v

Run the live integration test (requires initialized vector store):
    pytest test_chatbot.py -v -m integration

Run the interactive script directly:
    python test_chatbot.py
"""

import sys
import types
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

ACADEMIC_YEAR = "2"          # value stored in document metadata by DocumentProcessor
YEAR_LABEL    = "Year 2"

STACK_QUESTIONS = [
    "What is a stack?",
    "Explain stack push and pop operations",
    "What are the applications of a stack data structure?",
    "How is a stack implemented using an array?",
    "What is stack overflow and stack underflow?",
]


def _make_mock_pipeline(answer: str = "Mock answer about stack.", sources: list = None):
    """Return a lightweight mock RAGPipeline so unit tests need no real model."""

    class _MockPipeline:
        is_initialized = True

        def query(self, question, history=None, academic_year=None, **kwargs):
            return {
                "answer":  answer,
                "sources": sources or [
                    {
                        "content":  "A stack is a linear data structure...",
                        "metadata": {
                            "source":        "data_structures.pdf",
                            "academic_year": academic_year or ACADEMIC_YEAR,
                        },
                    }
                ],
                "context": "mock context",
            }

    return _MockPipeline()


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests (no real pipeline / vector store needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatbotUnitYear2Stack:
    """Unit tests — mocked pipeline, Year 2, stack questions."""

    def setup_method(self):
        self.rag = _make_mock_pipeline()

    # ── Basic response structure ───────────────────────────────────────────────

    def test_response_has_required_keys(self):
        result = self.rag.query(STACK_QUESTIONS[0], academic_year=ACADEMIC_YEAR)
        assert "answer"  in result, "Response must contain 'answer'"
        assert "sources" in result, "Response must contain 'sources'"

    def test_answer_is_non_empty_string(self):
        result = self.rag.query(STACK_QUESTIONS[0], academic_year=ACADEMIC_YEAR)
        assert isinstance(result["answer"], str)
        assert len(result["answer"].strip()) > 0, "Answer must not be empty"

    def test_sources_is_a_list(self):
        result = self.rag.query(STACK_QUESTIONS[0], academic_year=ACADEMIC_YEAR)
        assert isinstance(result["sources"], list)

    # ── Academic year propagation ─────────────────────────────────────────────

    def test_academic_year_passed_to_query(self):
        """The academic_year value must reach the pipeline."""
        received_years = []

        class _YearCapturePipeline:
            is_initialized = True
            def query(self, question, history=None, academic_year=None, **kwargs):
                received_years.append(academic_year)
                return {"answer": "ok", "sources": []}

        rag = _YearCapturePipeline()
        rag.query("What is a stack?", academic_year=ACADEMIC_YEAR)
        assert received_years == [ACADEMIC_YEAR], (
            f"Expected academic_year='{ACADEMIC_YEAR}', got {received_years}"
        )

    def test_source_metadata_contains_year(self):
        result = self.rag.query(STACK_QUESTIONS[0], academic_year=ACADEMIC_YEAR)
        if result["sources"]:
            year_in_source = result["sources"][0]["metadata"].get("academic_year")
            assert year_in_source == ACADEMIC_YEAR, (
                f"Source metadata academic_year should be '{ACADEMIC_YEAR}', got '{year_in_source}'"
            )

    # ── Stack-topic coverage ───────────────────────────────────────────────────

    @pytest.mark.parametrize("question", STACK_QUESTIONS)
    def test_all_stack_questions_return_answer(self, question):
        result = self.rag.query(question, academic_year=ACADEMIC_YEAR)
        assert result["answer"].strip(), f"Empty answer for: {question!r}"

    def test_answer_mentions_stack(self):
        result = self.rag.query("What is a stack?", academic_year=ACADEMIC_YEAR)
        assert "stack" in result["answer"].lower(), (
            "Answer to 'What is a stack?' should mention 'stack'"
        )

    # ── Conversation history ───────────────────────────────────────────────────

    def test_query_accepts_history(self):
        history = [
            {"role": "user",      "content": "What is a queue?"},
            {"role": "assistant", "content": "A queue is a FIFO structure."},
        ]
        result = self.rag.query(
            "How does a stack differ from a queue?",
            history=history,
            academic_year=ACADEMIC_YEAR,
        )
        assert result["answer"].strip()

    def test_empty_history_is_handled(self):
        result = self.rag.query(
            STACK_QUESTIONS[0],
            history=[],
            academic_year=ACADEMIC_YEAR,
        )
        assert result["answer"].strip()

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_arabic_stack_question(self):
        result = self.rag.query(
            "ما هو الـ stack وكيف يعمل؟",
            academic_year=ACADEMIC_YEAR,
        )
        assert result["answer"].strip()

    def test_no_year_filter_still_works(self):
        result = self.rag.query("What is a stack?")
        assert result["answer"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests  (marked — need a real, initialized vector store)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestChatbotIntegrationYear2Stack:
    """
    Integration tests — real RAGPipeline.
    Requires the vector store to be built before running.
    Run with:  pytest test_chatbot.py -v -m integration
    """

    @pytest.fixture(scope="class", autouse=True)
    def pipeline(self, request):
        from src.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        try:
            rag.vector_store_manager.load_vector_store()
            rag.is_initialized = True
        except Exception as exc:
            pytest.skip(f"Vector store not available — skipping integration tests: {exc}")
        request.cls.rag = rag

    def test_real_stack_question_returns_answer(self):
        result = self.rag.query(
            "What is a stack data structure?",
            academic_year=ACADEMIC_YEAR,
        )
        assert result["answer"].strip(), "Real pipeline returned an empty answer"

    def test_real_answer_not_error_message(self):
        result = self.rag.query(
            "Explain push and pop in a stack",
            academic_year=ACADEMIC_YEAR,
        )
        error_phrases = ["vector store not found", "please run initialization"]
        answer_lower = result["answer"].lower()
        for phrase in error_phrases:
            assert phrase not in answer_lower, f"Answer looks like an error: {result['answer']}"

    def test_real_sources_reference_year2(self):
        result = self.rag.query(
            "What is a stack?",
            academic_year=ACADEMIC_YEAR,
        )
        if result["sources"]:
            years = [s["metadata"].get("academic_year") for s in result["sources"]]
            assert any(y == ACADEMIC_YEAR for y in years), (
                f"Expected at least one Year {ACADEMIC_YEAR} source, got: {years}"
            )

    @pytest.mark.parametrize("question", STACK_QUESTIONS)
    def test_all_stack_questions_real_pipeline(self, question):
        result = self.rag.query(question, academic_year=ACADEMIC_YEAR)
        assert result["answer"].strip(), f"Empty answer for: {question!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive script  (python test_chatbot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_interactive():
    """Load the real pipeline and run the stack questions interactively."""
    print("=" * 65)
    print(f"  Chatbot Test — {YEAR_LABEL} | Stack Questions")
    print("=" * 65)

    from src.rag_pipeline import RAGPipeline
    rag = RAGPipeline()

    print("\n⚙  Loading vector store …")
    try:
        rag.vector_store_manager.load_vector_store()
        rag.is_initialized = True
        count = rag.vector_store_manager._safe_count()
        print(f"✓  Vector store loaded ({count} documents)\n")
    except Exception as exc:
        print(f"✗  Could not load vector store: {exc}")
        print("   Please run rag.initialize() first.")
        sys.exit(1)

    history = []

    for i, question in enumerate(STACK_QUESTIONS, 1):
        print(f"─" * 65)
        print(f"[{i}/{len(STACK_QUESTIONS)}] Question: {question}")
        print(f"         Academic Year: {YEAR_LABEL}")
        print()

        result = rag.query(
            question,
            history=history,
            academic_year=ACADEMIC_YEAR,
        )

        print(f"Answer:\n{result['answer']}\n")

        if result.get("sources"):
            print("Sources:")
            for src in result["sources"]:
                meta = src["metadata"]
                fname = meta.get("file_name", meta.get("source", "unknown"))
                year  = meta.get("academic_year", "?")
                print(f"  • {fname}  (year {year})")
        print()

        # Keep rolling history
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": result["answer"]})
        if len(history) > 10:
            history = history[-10:]

    print("=" * 65)
    print("  Test run complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_interactive()