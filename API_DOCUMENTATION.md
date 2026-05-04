# 🎓 EduEra API Documentation

Welcome to the EduEra Backend API. This system is a unified academic platform integrated with a RAG (Retrieval-Augmented Generation) AI engine.

## 🔑 Authentication
All APIs (except login) require a **JWT Bearer Token** in the header:
`Authorization: Bearer <your_access_token>`

### 1. Login
*   **Endpoint**: `POST /api/token/`
*   **Request Body**:
    ```json
    {
      "email": "student001@eduera.com",
      "password": "your_password"
    }
    ```
*   **Response**: Returns `access` and `refresh` tokens + user profile data.

### 2. Token Refresh
*   **Endpoint**: `POST /api/token/refresh/`
*   **Request Body**: `{"refresh": "<refresh_token>"}`

---

## 👨‍🎓 Student APIs
Base Path: `/api/student/`

### 1. Dashboard
*   **Endpoint**: `GET /dashboard/`
*   **Returns**: Active courses, recent announcements, and pending assignments.

### 2. Course Materials
*   **Endpoint**: `GET /courses/<course_id>/`
*   **Returns**: All files (PDF, PPTX) uploaded for this course.
*   **Download**: `GET /materials/<material_id>/download/`

### 3. AI Chatbot (The Core Feature)
*   **Endpoint**: `POST /chat/`
*   **Request Body**:
    ```json
    {
      "content": "What are the main topics in this course?",
      "course_id": 5,
      "conversation_id": null 
    }
    ```
*   **AI Smart Features**:
    *   **YouTube Support**: Include a URL in `content` to chat with the video.
    *   **Cumulative Access**: Students can ask about past (Completed) courses.
    *   **Context Memory**: Send a `conversation_id` to continue a previous chat.

---

## 👨‍🏫 Instructor (Professor/TA) APIs
Base Path: `/api/professor/` (or `/api/ta/`)

### 1. Upload Material (with AI Ingestion)
*   **Endpoint**: `POST /materials/`
*   **Format**: `multipart/form-data`
*   **Fields**: `file`, `title`, `course_offering`, `material_type`.
*   **AI Hook**: The system automatically chunks and indexes the file into the Vector DB immediately after upload.

### 2. Grading
*   **Endpoint**: `POST /submissions/<id>/grade/`
*   **Request Body**: `{"grade": 95, "feedback": "Great work!"}`

---

## 🛠️ Administrator APIs
Base Path: `/admin/`
Provides full CRUD (Create, Read, Update, Delete) via the Django Rest Framework router for:
*   `/users/`: Manage Students and Professors.
*   `/courses/`: Create and edit course descriptions.
*   `/departments/`: Manage faculty structures.

---

## 🧠 AI RAG Engine Specifications

### 1. Data Filtering (Security)
The AI does NOT search the whole database. It uses a **Pre-authorized Access List** calculated by the backend:
*   **Current Course**: `ACTIVE` status.
*   **Prerequisites**: `COMPLETED` status.
*   The LLM only sees documents matching these IDs.

### 2. YouTube Processing Pipeline
1.  **Stage 1 (API)**: Attempts to fetch official or auto-generated transcripts.
2.  **Stage 2 (Whisper)**: If subtitles are disabled, the AI "listens" to the audio and transcribes it on the GPU.
3.  **Caching**: Transcripts are saved in `ai_engine/data/transcript_cache/` for instant re-use.

### 3. Context Window Management
For very long lectures (>1 hour), the system automatically truncates the transcript to **8,000 tokens** to prevent API crashes and optimize costs/speed.

---

## 🚀 Deployment Notes
*   **GPU Requirement**: The server MUST have an NVIDIA GPU with `CUDA 12.1` for optimal performance.
*   **DB**: MySQL/MariaDB.
*   **Environment Variables**: Ensure `GROQ_API_KEY` and `EMBEDDING_DEVICE='cuda'` are set in `.env`.
