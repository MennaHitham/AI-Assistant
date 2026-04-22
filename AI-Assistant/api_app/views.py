import os
import time
import json
import logging
from django.http import FileResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from config.settings import RAW_DATA_DIR, SUPPORTED_EXTENSIONS
from models import (
    ChatRequest, ChatResponse, SourceInfo,
    YouTubeRequest, YouTubeResponse,
    RecommendationRequest, RecommendationResponse, VideoRecommendation,
    PresentationRequest, PresentationResponse,
    HealthResponse, InitializeResponse,
    UploadResponse, DocumentAskResponse, ImageUploadResponse,
)
from .services import rag

logger = logging.getLogger(__name__)

IMAGE_DIR = os.path.join("data", "presentation_images")
os.makedirs(IMAGE_DIR, exist_ok=True)


@api_view(['GET'])
def health_check(request):
    doc_count = 0
    if rag.is_initialized:
        try:
            doc_count = rag.vector_store_manager.vector_store._collection.count()
        except Exception:
            pass
    resp = HealthResponse(status="ok", initialized=rag.is_initialized, document_count=doc_count)
    return Response(resp.model_dump())


@api_view(['POST'])
def initialize_pipeline(request):
    try:
        rag.initialize()
        count = rag.vector_store_manager.vector_store._collection.count()
        resp = InitializeResponse(success=True, message="Pipeline initialized successfully", document_count=count)
        return Response(resp.model_dump())
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def chat(request):
    try:
        req_data = ChatRequest(**request.data)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    history = [msg.model_dump() for msg in req_data.history] if req_data.history else None
    try:
        result = rag.query(req_data.question, history=history)
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    sources = [SourceInfo(content=s["content"], metadata=s["metadata"]) for s in result.get("sources", [])]
    resp = ChatResponse(answer=result["answer"], sources=sources, presentation_path=result.get("presentation_path"))
    return Response(resp.model_dump())


@api_view(['POST'])
def process_youtube(request):
    try:
        req_data = YouTubeRequest(**request.data)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    data = rag.youtube_processor.process_url(req_data.url)
    if data is None:
        return Response({"detail": "Invalid or unsupported YouTube URL"}, status=status.HTTP_400_BAD_REQUEST)

    transcript = data.get("transcript")
    error = None
    if transcript and "[ERROR:" in str(transcript):
        error = transcript
        transcript = None

    answer = None
    sources = []

    if req_data.question and transcript:
        try:
            combined_query = f"{req_data.url} {req_data.question}"
            result = rag.query(combined_query)
            answer = result.get("answer")
            sources = result.get("sources", [])
        except Exception as e:
            logger.error(f"YouTube Q&A failed: {e}")
            error = str(e)
    elif req_data.question and not transcript:
        answer = "Could not extract transcript from this video, so I cannot answer the question."

    resp = YouTubeResponse(
        video_id=data.get("video_id", ""),
        title=data.get("title", "Unknown"),
        duration=data.get("duration", "Unknown"),
        transcript=transcript,
        answer=answer,
        sources=sources,
        error=error,
    )
    return Response(resp.model_dump())


@api_view(['POST'])
def get_recommendations(request):
    try:
        req_data = RecommendationRequest(**request.data)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    try:
        yt_results = rag.recommender.get_youtube_recommendations(req_data.topic, count=req_data.count)
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    videos = [VideoRecommendation(**v) for v in yt_results]
    resp = RecommendationResponse(topic=req_data.topic, youtube=videos)
    return Response(resp.model_dump())


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def create_presentation(request):
    presentation_data = request.data.get('presentation_data')
    if not presentation_data:
        return Response({"detail": "Missing presentation_data field"}, status=status.HTTP_400_BAD_REQUEST)

    images = request.FILES.getlist('images')
    try:
        images_by_filename = {}
        images_by_index = []
        if images:
            for image_file in images:
                save_path = os.path.join(IMAGE_DIR, image_file.name)
                with open(save_path, "wb") as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)
                images_by_filename[image_file.name] = save_path
                images_by_index.append(save_path)
                logger.info(f"Saved uploaded image: {image_file.name}")

        try:
            req_data = PresentationRequest.model_validate_json(presentation_data)
        except Exception as e:
            logger.error(f"Failed to parse presentation JSON: {e}")
            return Response({"detail": f"Invalid Presentation JSON structure: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        pptx_path = None
        if req_data.slides:
            slides_data = []
            final_image_paths = []
            for slide in req_data.slides:
                slides_data.append({"title": slide.title, "content": slide.content})
                slide_img_path = None
                if slide.image_index is not None:
                    if 0 <= slide.image_index < len(images_by_index):
                        slide_img_path = images_by_index[slide.image_index]
                    else:
                        logger.warning(f"Index {slide.image_index} out of range for slide '{slide.title}'")
                if not slide_img_path and slide.image_filename:
                    if slide.image_filename in images_by_filename:
                        slide_img_path = images_by_filename[slide.image_filename]
                    else:
                        stored_path = os.path.join(IMAGE_DIR, slide.image_filename)
                        if os.path.exists(stored_path):
                            slide_img_path = stored_path
                final_image_paths.append(slide_img_path)

            pptx_filename = f"{req_data.title.replace(' ', '_')}_{int(time.time())}.pptx"
            pptx_path = rag.presentation_maker.create_presentation(
                slides_data=slides_data, image_paths=final_image_paths, filename=pptx_filename
            )
        elif req_data.topic:
            question = f"Create a presentation about {req_data.topic} titled {req_data.title}"
            result = rag.query(question)
            pptx_path = result.get("presentation_path")
        else:
            return Response({"detail": "Either 'slides' or 'topic' must be provided."}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        logger.error(f"Presentation creation failed: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    if pptx_path:
        filename = os.path.basename(pptx_path)
        resp = PresentationResponse(success=True, message="Presentation created successfully", filename=filename, download_url=f"/presentation/download/{filename}")
        return Response(resp.model_dump())
    else:
        resp = PresentationResponse(success=False, message="Failed to generate presentation")
        return Response(resp.model_dump())


@api_view(['GET'])
def download_presentation(request, filename):
    file_path = os.path.join("presentations", filename)
    if not os.path.exists(file_path):
        return Response({"detail": "Presentation file not found"}, status=status.HTTP_404_NOT_FOUND)
    response = FileResponse(open(file_path, 'rb'), content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_document(request):
    file = request.FILES.get('file')
    if not file:
        return Response({"detail": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(file.name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return Response({"detail": f"Unsupported file type '{ext}'."}, status=status.HTTP_400_BAD_REQUEST)

    save_path = os.path.join(str(RAW_DATA_DIR), file.name)
    try:
        with open(save_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        rag.add_documents(save_path)
        resp = UploadResponse(success=True, message=f"File '{file.name}' uploaded and indexed", filename=file.name)
        return Response(resp.model_dump())
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_and_ask(request):
    file = request.FILES.get('file')
    question = request.data.get('question')
    if not file or not question:
        return Response({"detail": "File and question are required"}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(file.name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return Response({"detail": f"Unsupported file type '{ext}'."}, status=status.HTTP_400_BAD_REQUEST)

    save_path = os.path.join(str(RAW_DATA_DIR), file.name)
    try:
        with open(save_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)

        chunks = rag.document_processor.process_documents(save_path)
        if not chunks:
            resp = DocumentAskResponse(answer="I couldn't extract any text from this file, so I can't answer questions about its content.", sources=[], filename=file.name)
            return Response(resp.model_dump())

        rag.vector_store_manager.add_documents(chunks)
        result = rag.query(question, forced_documents=chunks)
        resp = DocumentAskResponse(answer=result["answer"], sources=result.get("sources", []), filename=file.name)
        return Response(resp.model_dump())
    except Exception as e:
        logger.error(f"Upload-and-ask failed: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


ALLOWED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    file = request.FILES.get('file')
    if not file:
        return Response({"detail": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return Response({"detail": f"Unsupported image type '{ext}'."}, status=status.HTTP_400_BAD_REQUEST)

    save_path = os.path.join(IMAGE_DIR, file.name)
    try:
        with open(save_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        resp = ImageUploadResponse(success=True, message=f"Image '{file.name}' uploaded successfully", filename=file.name, image_path=save_path)
        return Response(resp.model_dump())
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
