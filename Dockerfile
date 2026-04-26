# ─────────────────────────────────────────
# Dockerfile for HuggingFace Spaces
# ─────────────────────────────────────────

# Use official Python 3.11 slim image
# slim = smaller size, faster build
FROM python:3.11-slim

# ─── System dependencies ───────────────
# libgomp1 is required by FAISS
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Set working directory ─────────────
# All commands below run from /app
WORKDIR /app

# ─── Copy requirements first ───────────
# Docker caches this layer separately
# So pip install only reruns when
# requirements.txt actually changes
COPY backend/requirements.txt .

# ─── Install Python dependencies ───────
RUN pip install --no-cache-dir -r requirements.txt

# ─── Pre-download embedding model ──────
# We download the model at BUILD time
# so it's baked into the container
# This avoids slow downloads on first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ─── Copy all project files ────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# ─── Create storage folders ────────────
RUN mkdir -p ./backend/storage/uploads

# ─── Set working directory to backend ──
WORKDIR /app/backend

# ─── Expose port 7860 ──────────────────
# HuggingFace Spaces REQUIRES port 7860
EXPOSE 7860

# ─── Start FastAPI server ──────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]