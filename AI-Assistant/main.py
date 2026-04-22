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

    # Initialize RAG pipeline
    rag = RAGPipeline()

    # Try loading an existing vector store first
    try:
        logger.info("Attempting to load existing vector store...")
        rag.vector_store_manager.load_vector_store()

        count = rag.vector_store_manager._safe_count()

        if count == 0:
            raise ValueError("Vector store is empty, need to rebuild")

        rag.is_initialized = True
        print(f"\n✓ Loaded existing course materials database ({count} documents)")

    except Exception as e:
        logger.info(f"Need to initialize: {str(e)}")
        print("\n⚙ Processing course materials...")
        print(f"   Reading files from: {RAW_DATA_DIR} (year1 / year2 / year3 / year4)")

        try:
            rag.initialize()

            count = rag.vector_store_manager._safe_count()
            print(f"✓ Course materials processed successfully! ({count} documents)")

        except Exception as init_error:
            print(f"\n✗ Error during initialization: {str(init_error)}")
            print("\nPlease ensure:")
            print(f"1. Course files are placed inside the year folders:")
            print(f"   {RAW_DATA_DIR / 'year1'}")
            print(f"   {RAW_DATA_DIR / 'year2'}")
            print(f"   {RAW_DATA_DIR / 'year3'}")
            print(f"   {RAW_DATA_DIR / 'year4'}")
            print("2. Files are in supported formats: PDF, DOCX, PPTX, TXT, Images (PNG, JPG)")
            return

    # ================================================================== #
    # ★ التعديل الجديد: أخذ المواد المسجلة من الطالب ★
    # ================================================================== #
    print("\n" + "-" * 60)
    print("🎯 To improve search accuracy, please enter your enrolled courses.")
    print(fix_arabic_text("لتحسين دقة البحث، يرجى إدخال المواد المسجلة لديك."))
    print("   (Separate courses with a comma ',')")
    print(fix_arabic_text("   (افصل بين المواد بفاصلة ',')"))
    print("   Example: Data Structures, Algorithms, Databases")
    print(fix_arabic_text("   مثال: تراكيب بيانات, خوارزميات, قواعد بيانات"))
    print("-" * 60)
    
    courses_input = input(fix_arabic_text("📚 موادك الدراسية: ")).strip()
    
    user_courses = None
    if courses_input:
        # ننضف الأسماء ونحطهم في قائمة
        user_courses = [c.strip() for c in courses_input.split(',') if c.strip()]
        print(f"\n✓ Agent filtering activated for: {', '.join(user_courses)}")
        print(fix_arabic_text(f"✓ تم تفعيل فلتر المواد لـ: {', '.join(user_courses)}"))
    else:
        print("\n✓ No courses provided. The Agent will search across ALL course materials.")
        print(fix_arabic_text("✓ لم يتم إدخال مواد. سيقوم النظام بالبحث في جميع المواد المتاحة."))

    # ================================================================== #

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