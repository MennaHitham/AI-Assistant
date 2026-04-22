# 📚 RAG AI Teaching Assistant — API Documentation

Welcome to the API Documentation for the Course Material AI Assistant. This server exposes **10 REST endpoints** built with **Django REST Framework (DRF)**.

**Base URL**: `http://localhost:8000/`

---

## 🟢 1. System Endpoints

### 1.1 Health Check
**GET** `/health/`
Returns the status of the API, whether the vector store is initialized, and the total indexed documents.

**Response (JSON)**
```json
{
  "status": "ok",
  "initialized": true,
  "document_count": 1250
}
```

### 1.2 Initialize Pipeline
**POST** `/initialize/`
Reads all files in `data/raw/`, extracts text, and builds/rebuilds the underlying vector store.

**Response (JSON)**
```json
{
  "success": true,
  "message": "Pipeline initialized successfully",
  "document_count": 1250
}
```

---

## 💬 2. Core AI Chat and Querying

### 2.1 Unified Chat
**POST** `/chat/`
The main AI endpoint for Q&A. It automatically detects if the question is a normal query, a YouTube URL analysis request, a recommendation request, or a presentation topic request.

**Request Body (JSON)**
```json
{
  "question": "What is the difference between TCP and UDP?",
  "history": [
    {
      "role": "user",
      "content": "Tell me about networking."
    },
    {
      "role": "assistant",
      "content": "Networking involves..."
    }
  ]
}
```
*Note: `history` is optional.*

**Response (JSON)**
```json
{
  "answer": "TCP is connection-oriented, while UDP is connectionless...",
  "sources": [
    {
      "content": "- TCP uses a three-way handshake...",
      "metadata": {"source": "data/raw/networking_lecture.pdf", "page": 5}
    }
  ],
  "presentation_path": null
}
```

---

## 📄 3. Document Management

### 3.1 Upload Document
**POST** `/documents/upload/`
Uploads a course file (PDF, DOCX, PPTX, TXT, images) to the system and indexes it into the vector store.

**Request Body (Multipart/Form-Data)**
- `file`: The course file to upload (Binary)

**Response (JSON)**
```json
{
  "success": true,
  "message": "File 'lecture.pdf' uploaded and indexed",
  "filename": "lecture.pdf"
}
```

### 3.2 Instant Document Q&A (Upload & Ask)
**POST** `/documents/ask/`
Uploads a document and **immediately** asks a question about it, restricting the AI’s context strictly to the uploaded file.

**Request Body (Multipart/Form-Data)**
- `file`: The document (Binary)
- `question`: "Summarize this exam" (Text string)

**Response (JSON)**
```json
{
  "answer": "The exam covers variables, loops, and pointers.",
  "sources": [...],
  "filename": "midterm_exam.pdf"
}
```

---

## 🎥 4. YouTube Analysis & Recommendations

### 4.1 Process YouTube Video
**POST** `/youtube/process/`
Analyzes a YouTube URL. If a `question` is provided, the AI answers it based on the video's transcript.

**Request Body (JSON)**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "question": "What is the main topic of this music video?"
}
```
*Note: `question` is optional. If missing, the transcript is returned natively.*

**Response (JSON)**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "duration": "3:32",
  "transcript": "...",
  "answer": "The main topic is a song about commitment...",
  "sources": [],
  "error": null
}
```

### 4.2 YouTube Course Recommendations
**POST** `/recommendations/`
Fetches educational video recommendations relating to your course material.

**Request Body (JSON)**
```json
{
  "topic": "Machine Learning Transformers",
  "count": 5
}
```

**Response (JSON)**
```json
{
  "topic": "Machine Learning Transformers",
  "youtube": [
    {
      "title": "Attention is All You Need Explained",
      "url": "https://www.youtube.com/watch?v=...",
      "duration": "14:20"
    }
  ]
}
```

---

## 📊 5. Presentation Maker

### 5.1 Upload Presentation Image
**POST** `/images/upload/`
Uploads an image specifically to be used in manually created presentations.

**Request Body (Multipart/Form-Data)**
- `file`: The image file (PNG, JPG, JPEG, WEBP)

**Response (JSON)**
```json
{
  "success": true,
  "message": "Image 'diagram.png' uploaded successfully",
  "filename": "diagram.png",
  "image_path": "data/presentation_images/diagram.png"
}
```

### 5.2 Create Presentation
**POST** `/presentation/create/`
Generates a PowerPoint (`.pptx`) presentation.
It operates in **two modes**:
1. **AI Mode**: Provide a `topic` and the AI writes the slides.
2. **Manual Mode**: Provide structured `slides` with exact titles and bullet points.

**Important for Manual Mode**: You can attach images directly to this request (`images` field) and map them to slides either by their positional index (`image_index`) or filename (`image_filename`).

**Request Body (Multipart/Form-Data)**
- `presentation_data`: (Required JSON String)
- `images`: (Optional, multiple files)

**Example `presentation_data` JSON for AI Mode:**
```json
{
  "title": "Intro to AI",
  "topic": "Artificial Intelligence History"
}
```

**Example `presentation_data` JSON for Manual Mode (with Images):**
```json
{
  "title": "Machine Learning Models",
  "slides": [
    {
      "title": "Neural Networks",
      "content": ["Artificial neurons", "Layers of abstraction"],
      "image_index": 0 
    },
    {
      "title": "Transformers",
      "content": ["Self-attention mechanisms"],
      "image_filename": "transformer_architecture.png"
    }
  ]
}
```

**Response (JSON)**
```json
{
  "success": true,
  "message": "Presentation created successfully",
  "filename": "Machine_Learning_Models_1700000000.pptx",
  "download_url": "/presentation/download/Machine_Learning_Models_1700000000.pptx"
}
```

### 5.3 Download Presentation
**GET** `/presentation/download/<filename>/`
Downloads a previously generated `.pptx` file based on the filename returned from the create endpoint.

**Response**
Returns the binary `.pptx` file with `Content-Disposition: attachment`.
