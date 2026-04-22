from src.rag_pipeline import RAGPipeline
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler("test_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def test_presentation_maker():
    print("Testing Presentation Maker Feature...")
    rag = RAGPipeline()
    
    # Mock initialization if it takes too long, but we need it for real content
    # For now, let's try a real query that will trigger a presentation
    # If it fails to load, we'll initialize
    try:
        rag.vector_store_manager.load_vector_store()
        rag.is_initialized = True
    except:
        print("Rebuilding vector store for test...")
        rag.initialize()

    question = "Create a presentation about Machine Learning based on the course materials"
    print(f"\nQUERY: {question}")
    
    try:
        result = rag.query(question)
        print("\nANSWER:")
        print(result['answer'])
        
        if 'presentation_path' in result:
            path = result['presentation_path']
            print(f"\n✓ Presentation successfully created at: {path}")
            if os.path.exists(path):
                print(f"✓ File confirmed on disk: {os.path.abspath(path)}")
            else:
                print("✗ File NOT found on disk!")
        else:
            print("\n✗ Presentation path not found in result.")
            
    except Exception as e:
        print(f"\nCRITICAL ERROR in presentation test: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_presentation_maker()
