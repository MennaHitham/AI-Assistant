FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# tesseract-ocr-ara is crucial for Arabic support
# poppler-utils is needed for pdf2image
# curl for healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-ara \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the API port
EXPOSE 8000

# Run the Django server
CMD ["python", "AI-Assistant/manage.py", "runserver", "0.0.0.0:8000"]
