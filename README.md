# 🎓 Course Material AI Assistant (مساعد المواد الدراسية الذكي)

A powerful **RAG-based** (Retrieval-Augmented Generation) AI teaching assistant that helps students and educators interact with multi-format course materials through a unified chat interface. The system supports **Arabic and English**, processes documents, images, and YouTube videos, and exposes all features as **REST APIs** for front-end integration.

---

## 🚀 Features

### Core AI Capabilities
- **Smart Q&A** — Ask questions about your course materials and get AI-generated answers with source references.
- **YouTube Video Analysis** — Paste a YouTube URL and ask questions. The system extracts the transcript and answers from the video content.
- **Recommendations** — Ask for learning recommendations and get curated YouTube video suggestions.
- **Presentation Generation** — Ask the AI to create a PowerPoint presentation on any topic from your course materials.
- **Document & Image Understanding** — Upload PDFs, DOCX, PPTX, images, and more. The system extracts text (including OCR for scanned documents) and answers questions.
- **Conversation History** — The system remembers previous messages within a session for follow-up questions.

### Technical Highlights
- **Multilingual Support** — Fully optimized for Arabic and English using `paraphrase-multilingual-MiniLM-L12-v2` embeddings.
- **Hybrid Retrieval** — Combines dense vector search (ChromaDB) with BM25 sparse retrieval, followed by CrossEncoder reranking for best accuracy.
- **Groq API** — Powered by Llama 3.3 70B Versatile via [Groq](https://groq.com/) for fast, high-quality responses.
- **OCR** — Tesseract-based text extraction from scanned PDFs and standalone images (supports Arabic + English).
- **Smart Filtering** — Automatically filters out assessment-style content (MCQs) to provide cleaner context.
- **Academic Year Filtering** — Course materials are organized by year (year1–year4) and can be queried with a year filter.
- **REST API** — Full Django Rest Framework (DRF) server for seamless front-end integration.

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or 3.11 |
| **Groq API Key** | Sign up at [groq.com](https://groq.com/) and get a free API key |
| **Tesseract OCR** | **Windows**: Install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) (include Arabic language data). **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-ara` |
| **Poppler** *(optional, for PDF images)* | **Linux**: `sudo apt install poppler-utils`. **Windows**: Download poppler and add `bin/` to PATH. |
| **ffmpeg** *(optional, for YouTube fallback)* | Required only if YouTube videos don't have transcripts and need Whisper transcription. |

---

## 🛠️ Installation (Step-by-Step)

### 1. Clone the Repository

```bash
git clone https://github.com/MennaHitham/AI-Assistant.git
cd AI-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
# Windows (PowerShell):
venv\Scripts\activate

# Windows (CMD):
venv\Scripts\activate.bat

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This will install PyTorch (CPU version), sentence-transformers, Django, Django REST Framework, and all other dependencies. The first run will also download the embedding model (~130MB).

### 4. Set Up Your Groq API Key

Create a `.env` file inside the `AI-Assistant/` directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

> You can get a free API key from [console.groq.com](https://console.groq.com/).

### 5. Add Your Course Materials

Place your course files inside the year folders under `AI-Assistant/data/`:

```
AI-Assistant/
  data/
    year1/          ← First year course files
      lecture1.pdf
      notes.docx
    year2/          ← Second year course files
      slides.pptx
    year3/          ← Third year course files
      diagram.png
    year4/          ← Fourth year course files
      textbook.pdf
```

**Supported formats:** PDF, DOCX, PPTX, TXT, PNG, JPG, JPEG

> Files are automatically tagged with their academic year during processing, enabling year-specific filtering when querying.

### 6. (Optional) Add Presentation Images

If you want images in your generated presentations, place them in:

```
AI-Assistant/
  data/
    presentation_images/    ← Put images here
      slide1.png
      slide2.jpg
```

---

## 🚀 Running the Project

### Option A: Run the API Server (Recommended for front-end integration)

```bash
cd AI-Assistant
python manage.py runserver 0.0.0.0:8000
```

The server will:
1. Automatically load the existing vector store on startup (if available).
2. Serve the API at **http://localhost:8000**
3. Provide a Browsable API at the endpoint URLs (if accessed via browser)

> **First time?** After starting the server, call `POST /initialize` to process your course materials and build the vector store. This only needs to be done once (or when you add new files).

### Option B: Run the CLI (Terminal-based chat)

```bash
cd AI-Assistant
python main.py
```

---

## 🌐 API Reference

The API server provides the following endpoints. The primary endpoint is `/chat` — it handles **all features automatically** through intelligent intent detection. The other endpoints are available for direct access if the front-end needs them.

### Primary Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | **Unified chat** — send any question and the AI automatically detects intent (Q&A, YouTube, recommendations, presentation) |

The `/chat` endpoint accepts:
```json
{
  "question": "Your question here (can include a YouTube URL)",
  "academic_year": "2",
  "history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
}
```

> `academic_year` is optional. When provided (e.g. `"1"`, `"2"`, `"3"`, `"4"`), retrieval is filtered to that year's materials only. When omitted, the system searches across all years.

**How intent detection works in `/chat`:**

| What the user types | Auto-detected intent | What happens |
|---|---|---|
| `"What is a stack?"` | Q&A | Searches course materials and answers |
| `"https://youtube.com/watch?v=xxx What is this about?"` | YouTube Q&A | Extracts transcript, answers from video |
| `"Recommend me courses about Python"` | Recommendation | Fetches YouTube video recommendations |
| `"Create a presentation about linked lists"` | Presentation | Generates a .pptx file |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns API status and document count |
| `POST` | `/initialize` | Process course materials from all year folders and build the vector store |

### Utility Endpoints (Direct access)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/youtube/process` | Process a YouTube URL directly. Accepts optional `question` field |
| `POST` | `/recommendations` | Get YouTube recommendations for a topic |
| `POST` | `/presentation/create` | Generate a PowerPoint presentation for a topic |
| `GET` | `/presentation/download/{filename}` | Download a generated .pptx file |
| `POST` | `/documents/upload` | Upload a course material file (indexes it into the vector store) |
| `POST` | `/documents/ask` | Upload a file AND ask a question about it in one call |
| `POST` | `/images/upload` | Upload an image for use in generated presentations |

---

## 🐳 Docker Support

### Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build the container with all dependencies (including Tesseract OCR).
- Start the API server on **port 8000**.
- Pass your `GROQ_API_KEY` from the `.env` file automatically.

### Run with Docker only

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here rag-assistant
```

---

## 📁 Project Structure

```
RAG-AI-Teaching-Assistant-/
├── AI-Assistant/
│   ├── manage.py                 # Django management script
│   ├── config_proj/              # Django site configuration
│   ├── api_app/                  # Django REST API application
│   ├── models.py                 # Pydantic request/response schemas
│   ├── main.py                   # CLI entry point
│   ├── .env                      # Your Groq API key (not committed to git)
│   ├── config/
│   │   └── settings.py           # All configuration (paths, models, thresholds)
│   ├── data/
│   │   ├── year1/                # ← First year course materials
│   │   ├── year2/                # ← Second year course materials
│   │   ├── year3/                # ← Third year course materials
│   │   ├── year4/                # ← Fourth year course materials
│   │   ├── processed/            # Vector store & chunk cache (auto-generated)
│   │   │   ├── chroma_db/        # ChromaDB vector store
│   │   │   └── processed_chunks.jsonl  # Cached document chunks
│   │   └── presentation_images/  # ← Images for generated presentations
│   ├── src/
│   │   ├── rag_pipeline.py       # Main orchestrator (intent detection + query routing)
│   │   ├── document_processor.py # Loads & chunks documents (PDF, DOCX, PPTX, images)
│   │   ├── youtube_processor.py  # YouTube transcript extraction (API + Whisper fallback)
│   │   ├── generator.py          # LLM answer generation via Groq API
│   │   ├── recommender.py        # YouTube video recommendations
│   │   ├── presentation_maker.py # PowerPoint (.pptx) generation
│   │   ├── vector_store.py       # ChromaDB vector store management
│   │   ├── retriever.py          # Hybrid retrieval (dense + BM25 + reranking)
│   │   └── embeddings.py         # Sentence-transformer embedding manager
│   ├── utils/
│   │   └── helpers.py            # Arabic text display + formatting utilities
│   └── presentations/            # Generated presentations output
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Configuration

All settings are in `AI-Assistant/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | from `.env` | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | The LLM model to use |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model (Arabic + English) |
| `EMBEDDING_DEVICE` | `cpu` | Set to `cuda` if you have a GPU |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Number of documents to retrieve per query |
| `ENABLE_OCR` | `True` | Enable/disable image text extraction |
| `SIMILARITY_THRESHOLD` | `0.45` | Minimum similarity score for retrieval |

---

## 🧪 Quick Test

After starting the API server, you can test it immediately:

```bash
# Health check
curl http://localhost:8000/health

# Initialize (first time only — processes all year folders)
curl -X POST http://localhost:8000/initialize

# Ask a question (all years)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a linked list?"}'

# Ask a question filtered to Year 2
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a stack?", "academic_year": "2"}'

# Ask about a YouTube video
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "https://youtube.com/watch?v=VIDEO_ID Explain what this video covers"}'

# Get recommendations
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Recommend me courses about data structures"}'
```

Or open the endpoints in your browser to view the DRF browsable API.