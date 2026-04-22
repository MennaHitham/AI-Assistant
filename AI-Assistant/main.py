from src.rag_pipeline import RAGPipeline
from utils.helpers import format_sources, print_divider, fix_arabic_text
from config.settings import RAW_DATA_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the RAG chatbot."""
    print("=" * 60)
    print("Course Material AI Assistant")
    print(fix_arabic_text("مساعد المواد الدراسية الذكي"))
    print("=" * 60)

    # Initialize RAG
    rag = RAGPipeline()
    try:
        # Check if store exists and has data
        if rag.vector_store_manager.store_exists():
            logger.info("Existing vector store found. Loading...")
            rag.vector_store_manager.load_vector_store()
            count = rag.vector_store_manager._safe_count()
            
            if count > 0:
                rag.is_initialized = True
                print(f"\n✓ Loaded existing course materials database ({count} documents)")
            else:
                logger.info("Vector store is empty. Initializing...")
                rag.initialize()
        else:
            logger.info("No vector store found. Starting initialization...")
            rag.initialize()

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        print(f"\n✗ Error: Could not load data. Ensure you are using the correct Python environment.")
        return

    # Interactive query loop
    history = []
    user_courses = None

    print("\n" + "=" * 60)
    print("You can now ask questions about your course materials or provide a YouTube URL!")
    print(fix_arabic_text("يمكنك الآن طرح أسئلة حول موادك الدراسية أو تزويدنا برابط يوتيوب!"))
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    # Interactive query loop
    history = []
    while True:
        try:
            question = input("\n🎓 Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Course Material AI Assistant!")
                print(fix_arabic_text("شكراً لاستخدامك مساعد المواد الدراسية!"))
                break

            if not question:
                continue

            print_divider()
            print("🔍 Agent is thinking (Rewriting -> Filtering -> Evaluating -> Routing)...\n")

            # ★ التعديل الجديد: تمرير الـ user_courses للـ query ★
            result = rag.query(
                question, 
                history=history, 
                user_courses=user_courses  # هنا بنبعت المواد عشان الـ Agent يفلتر بيهم
            )

            # Display answer
            print("📖 Answer:")
            print(fix_arabic_text(result['answer']))

            # Update history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result['answer']})

            if len(history) > 10:  # 5 user + 5 assistant
                history = history[-10:]

            # Display sources (لو موجودة، ومفيش ديسكلايمر يعني الجواب من المحاضرات)
            if result.get('sources'):
                print(fix_arabic_text(format_sources(result['sources'])))
            
            # لو طلع بريزنتيشن
            if result.get('presentation_path'):
                print(f"\n📎 Presentation saved at: {result['presentation_path']}")

            print_divider()

        except KeyboardInterrupt:
            print("\n\nThank you for using the Course Material AI Assistant!")
            print(fix_arabic_text("شكراً لاستخدامك مساعد المواد الدراسية!"))
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print(f"\n✗ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()