"""
FastAPI REST API for the RAG AI Teaching Assistant.

Exposes all AI features as HTTP endpoints so the front-end can
integrate with the AI backend.

Run locally:
    cd AI-Assistant
    python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import logging
import shutil
import time
import json
from typing import List

from pydantic import Json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.rag_pipeline import RAGPipeline
from config.settings import RAW_DATA_DIR, SUPPORTED_EXTENSIONS
from models import (
    ChatRequest, ChatResponse, SourceInfo,
    YouTubeRequest, YouTubeResponse,
    RecommendationRequest, RecommendationResponse, VideoRecommendation,
    PresentationRequest, PresentationResponse,
    HealthResponse, InitializeResponse,
    UploadResponse, DocumentAskResponse, ImageUploadResponse,
)

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for presentation images
IMAGE_DIR = os.path.join("data", "presentation_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ─── App Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="RAG AI Teaching Assistant API",
    description="REST API for the Course-Material AI Assistant — "
                "Q&A, YouTube processing, recommendations, presentations.",
    version="1.0.0",
)

# Allow the front-end (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Pipeline (singleton) ───────────────────────────────────
rag = RAGPipeline()


# ─── Startup Event ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Try to load an existing vector store on startup."""
    try:
        rag.vector_store_manager.load_vector_store()
        count = rag.vector_store_manager.vector_store._collection.count()
        if count > 0:
            rag.is_initialized = True
            logger.info(f"Vector store loaded on startup ({count} documents)")
        else:
            logger.info("Vector store is empty — call /initialize to build it")
    except Exception as e:
        logger.info(f"No existing vector store found: {e}")


# ═══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════


# ─── 1. Health Check ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return API status and vector store document count."""
    doc_count = 0
    if rag.is_initialized:
        try:
            doc_count = rag.vector_store_manager.vector_store._collection.count()
        except Exception:
            pass
    return HealthResponse(
        status="ok",
        initialized=rag.is_initialized,
        document_count=doc_count,
    )


# ─── 2. Initialize Pipeline ───────────────────────────────────────
@app.post("/initialize", response_model=InitializeResponse, tags=["System"])
async def initialize_pipeline():
    """Process course materials from data/raw and build the vector store."""
    try:
        rag.initialize()
        count = rag.vector_store_manager.vector_store._collection.count()
        return InitializeResponse(
            success=True,
            message="Pipeline initialized successfully",
            document_count=count,
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── 3. Chat / Q&A ────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main Q&A endpoint.

    Send a question (optionally containing a YouTube URL) and optional
    conversation history.  The pipeline will:
      - Retrieve relevant course material
      - Process YouTube transcript (if URL detected)
      - Detect intent (recommendation / presentation / Q&A)
      - Return an AI-generated answer with sources
    """
    # Convert Pydantic history to plain dicts as the pipeline expects
    history = None
    if request.history:
        history = [msg.model_dump() for msg in request.history]

    try:
        result = rag.query(request.question, history=history)
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        SourceInfo(content=s["content"], metadata=s["metadata"])
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        presentation_path=result.get("presentation_path"),
    )


# ─── 4. YouTube Processing & Q&A ──────────────────────────────────
@app.post("/youtube/process", response_model=YouTubeResponse, tags=["YouTube"])
async def process_youtube(request: YouTubeRequest):
    """
    Process a YouTube video URL.

    - **Without question**: returns transcript + metadata only.
    - **With question**: the AI answers the question from the video transcript.
    """
    data = rag.youtube_processor.process_url(request.url)

    if data is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported YouTube URL")

    transcript = data.get("transcript")
    error = None
    if transcript and "[ERROR:" in str(transcript):
        error = transcript
        transcript = None

    answer = None
    sources = []

    # If the user asked a question, feed the transcript to the LLM
    if request.question and transcript:
        try:
            # Build a combined query so the pipeline processes both
            combined_query = f"{request.url} {request.question}"
            result = rag.query(combined_query)
            answer = result.get("answer")
            sources = result.get("sources", [])
        except Exception as e:
            logger.error(f"YouTube Q&A failed: {e}")
            error = str(e)
    elif request.question and not transcript:
        answer = "Could not extract transcript from this video, so I cannot answer the question."

    return YouTubeResponse(
        video_id=data.get("video_id", ""),
        title=data.get("title", "Unknown"),
        duration=data.get("duration", "Unknown"),
        transcript=transcript,
        answer=answer,
        sources=sources,
        error=error,
    )


# ─── 5. Recommendations ───────────────────────────────────────────
@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """Get YouTube video recommendations for a given topic."""
    try:
        yt_results = rag.recommender.get_youtube_recommendations(
            request.topic, count=request.count
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    videos = [VideoRecommendation(**v) for v in yt_results]
    return RecommendationResponse(topic=request.topic, youtube=videos)


# ─── 6. Presentation — Create ─────────────────────────────────────
@app.post("/presentation/create", response_model=PresentationResponse, tags=["Presentation"])
async def create_presentation(
    presentation_data: str = Form(..., description="The presentation structure inside a JSON object"),
    images: List[UploadFile] = File(None, description="Image files to use in the presentation (referenced by index or filename)")
):
    """
    Generate a PowerPoint presentation with integrated image uploads.
    
    The 'presentation_data' field expects a JSON object string.
    Each slide can reference an image by 'image_index' (logical order) 
    or 'image_filename' (variable name).
    """
    try:
        # 0. Save uploaded images and keep track of them
        images_by_filename = {}
        images_by_index = []
        
        if images:
            for image_file in images:
                save_path = os.path.join(IMAGE_DIR, image_file.filename)
                with open(save_path, "wb") as f:
                    content = await image_file.read()
                    f.write(content)
                images_by_filename[image_file.filename] = save_path
                images_by_index.append(save_path)
                logger.info(f"Saved uploaded image: {image_file.filename}")

        # 1. Parse the JSON data manually for better error reporting
        try:
            request = PresentationRequest.model_validate_json(presentation_data)
        except Exception as e:
            logger.error(f"Failed to parse presentation JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid Presentation JSON structure: {str(e)}")

        # --- Mode 1: Manual Structured Input ---
        if request.slides:
            logger.info(f"Creating manual presentation: '{request.title}' with {len(request.slides)} slides")
            
            slides_data = []
            final_image_paths = []
            
            for slide in request.slides:
                slides_data.append({
                    "title": slide.title,
                    "content": slide.content
                })
                
                # Resolve image using BOTH methods (Index prioritized, then Filename)
                slide_img_path = None
                
                # Method A: Index mapping
                if slide.image_index is not None:
                    if 0 <= slide.image_index < len(images_by_index):
                        slide_img_path = images_by_index[slide.image_index]
                    else:
                        logger.warning(f"Index {slide.image_index} out of range for slide '{slide.title}'")
                
                # Method B: Filename mapping (if index didn't work)
                if not slide_img_path and slide.image_filename:
                    if slide.image_filename in images_by_filename:
                        slide_img_path = images_by_filename[slide.image_filename]
                    else:
                        # Final fallback: Check existing storage
                        stored_path = os.path.join(IMAGE_DIR, slide.image_filename)
                        if os.path.exists(stored_path):
                            slide_img_path = stored_path
                
                final_image_paths.append(slide_img_path)

            # Generate PPTX directly
            pptx_filename = f"{request.title.replace(' ', '_')}_{int(time.time())}.pptx"
            pptx_path = rag.presentation_maker.create_presentation(
                slides_data=slides_data,
                image_paths=final_image_paths,
                filename=pptx_filename
            )
            
        # --- Mode 2: AI Generated from Topic ---
        elif request.topic:
            logger.info(f"Creating AI presentation for topic: '{request.topic}'")
            question = f"Create a presentation about {request.topic} titled {request.title}"
            result = rag.query(question)
            pptx_path = result.get("presentation_path")
        
        else:
            raise HTTPException(status_code=400, detail="Either 'slides' or 'topic' must be provided.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Presentation creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    if pptx_path:
        filename = os.path.basename(pptx_path)
        return PresentationResponse(
            success=True,
            message="Presentation created successfully",
            filename=filename,
            download_url=f"/presentation/download/{filename}",
        )
    else:
        return PresentationResponse(success=False, message="Failed to generate presentation")


# ─── 7. Presentation — Download ───────────────────────────────────
@app.get("/presentation/download/{filename}", tags=["Presentation"])
async def download_presentation(filename: str):
    """Download a previously generated .pptx file."""
    file_path = os.path.join("presentations", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Presentation file not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


# ─── 8. Upload Course Documents ───────────────────────────────────
@app.post("/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a course material file (PDF, DOCX, PPTX, TXT, or image).

    The file is saved to data/raw and then indexed into the vector store.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    save_path = os.path.join(str(RAW_DATA_DIR), file.filename)
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Add document to vector store
        rag.add_documents(save_path)

        return UploadResponse(
            success=True,
            message=f"File '{file.filename}' uploaded and indexed",
            filename=file.filename,
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── 8b. Upload Document & Ask a Question ─────────────────────────
@app.post("/documents/ask", response_model=DocumentAskResponse, tags=["Documents"])
async def upload_and_ask(
    file: UploadFile = File(...),
    question: str = Form(..., description="Question to ask about this file"),
):
    """
    Upload a file (PDF, DOCX, image, etc.) AND ask a question about it.

    The system will:
    1. Save and index the file.
    2. Use the AI to answer your question based on the file content.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    save_path = os.path.join(str(RAW_DATA_DIR), file.filename)
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 1. Process and Index the new document
        logger.info(f"Indexing new document for immediate Q&A: {file.filename}")
        chunks = rag.document_processor.process_documents(save_path)
        
        if not chunks:
            logger.warning(f"No text extracted from {file.filename}. OCR might have failed.")
            return DocumentAskResponse(
                answer="I couldn't extract any text from this file, so I can't answer questions about its content.",
                sources=[],
                filename=file.filename
            )
        
        # Log the first bit of text so the user can verify OCR in the terminal
        sample_text = chunks[0].page_content[:200].replace("\n", " ")
        logger.info(f"Extracted text sample: {sample_text}...")

        # Add to vector store
        rag.vector_store_manager.add_documents(chunks)

        # 2. Query the pipeline
        # We pass the extracted chunks as 'forced_documents'. 
        # This bypasses the general vector search and ensures the AI 
        # answers ONLY using the content of the file just uploaded.
        logger.info(f"Querying with forced context from {file.filename}")
        
        result = rag.query(question, forced_documents=chunks)

        return DocumentAskResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            filename=file.filename,
        )
    except Exception as e:
        logger.error(f"Upload-and-ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── 9. Upload Presentation Images ────────────────────────────────
ALLOWED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


@app.post("/images/upload", response_model=ImageUploadResponse, tags=["Images"])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image to be used in generated presentations.

    Images are saved to data/presentation_images/ and automatically
    picked up by the presentation generator.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{ext}'. Allowed: {ALLOWED_IMAGE_EXTENSIONS}",
        )

    save_path = os.path.join(IMAGE_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return ImageUploadResponse(
            success=True,
            message=f"Image '{file.filename}' uploaded successfully",
            filename=file.filename,
            image_path=save_path,
        )
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
