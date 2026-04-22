from src.rag_pipeline import RAGPipeline
import logging
import os

# Configure logging to see output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def run_presentation_generation():
    print("🚀 Starting Presentation Generation Process...")
    rag = RAGPipeline()
    
    # Initialize or load vector store
    try:
        print("📂 Loading vector store...")
        rag.vector_store_manager.load_vector_store()
        rag.is_initialized = True
    except Exception as e:
        print(f"⚙️ Rebuilding vector store: {e}")
        rag.initialize()

    # Query specifically for a presentation
    question = "Create a presentation about Data Structures (Stacks, Queues, and Linked Lists) based on the course materials"
    print(f"\n📝 QUERY: {question}")
    
    try:
        result = rag.query(question)
        print("\n🤖 AI RESPONSE:")
        print(result['answer'])
        
        if 'presentation_path' in result:
            path = result['presentation_path']
            print(f"\n✅ Presentation successfully created at: {path}")
            if os.path.exists(path):
                print(f"⭐ File verified at: {os.path.abspath(path)}")
            else:
                print("❌ File NOT found on disk despite success message!")
        else:
            print("\n❌ Presentation path not found in result dictionary.")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    run_presentation_generation()
