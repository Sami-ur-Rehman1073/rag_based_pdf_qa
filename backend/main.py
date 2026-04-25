import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from utils.pdf_utils import extract_text_from_pdf
from utils.text_utils import split_text_into_chunks
from services.embedding_service import embed_chunks, embed_question
from services.faiss_service import save_faiss_index, search_faiss_index
from models.schemas import MessageResponse, QuestionRequest, AnswerResponse, SourceChunk

# -----------------------------------------------
# Storage folders
# -----------------------------------------------
UPLOAD_DIR = "../storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------------------
# FastAPI app
# -----------------------------------------------
app = FastAPI(
    title="RAG PDF Q&A System",
    description="Upload a PDF and ask questions about it — fully local, no paid APIs.",
    version="1.0.0"
)

# -----------------------------------------------
# CORS Middleware
# -----------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# Serve frontend/index.html at root URL
# Mount the frontend folder as static files
#
# Why: Opening index.html via file:// causes
# browser security blocks on fetch() calls.
# Serving via FastAPI puts both on same origin.
# -----------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)

# -----------------------------------------------
# Serve index.html at http://127.0.0.1:8000/app
# -----------------------------------------------
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


# -----------------------------------------------
# Health Check API (JSON)
# -----------------------------------------------
@app.get("/health", response_class=JSONResponse)
async def root():
    return {
        "status": "running",
        "message": "RAG PDF Q&A backend is live!",
        "endpoints": {
            "frontend"     : "GET  /",
            "upload_pdf"   : "POST /upload-pdf",
            "ask_question" : "POST /ask-question",
            "docs"         : "GET  /docs"
        }
    }


# -----------------------------------------------
# POST /upload-pdf
# -----------------------------------------------
@app.post("/upload-pdf", response_model=MessageResponse)
async def upload_pdf(file: UploadFile = File(...)):

    # ---- Validate file type ----
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # ---- Save uploaded file ----
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    # ---- Extract text ----
    try:
        extracted_text = extract_text_from_pdf(save_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text: {str(e)}"
        )

    # ---- Split into chunks ----
    try:
        chunks = split_text_into_chunks(
            text=extracted_text,
            chunk_size=500,
            overlap=100
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to chunk text: {str(e)}"
        )

    # ---- Embed chunks ----
    try:
        embeddings = embed_chunks(chunks)
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )

    # ---- Save to FAISS ----
    try:
        save_faiss_index(embeddings, chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save FAISS index: {str(e)}"
        )

    return MessageResponse(
        message=f"PDF fully processed and stored. {len(chunks)} chunks indexed.",
        filename=file.filename,
        chunks_created=len(chunks)
    )


# -----------------------------------------------
# POST /ask-question
# -----------------------------------------------
@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    # ---- Validate question ----
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # ---- Embed the question ----
    try:
        question_embedding = embed_question(request.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed question: {str(e)}"
        )

    # ---- Search FAISS ----
    try:
        results = search_faiss_index(
            question_embedding=question_embedding,
            top_k=3
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search index: {str(e)}"
        )

    # ---- Handle no results ----
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found. Try rephrasing your question."
        )

    # ---- Build response ----
    best_answer = results[0]["text"]

    source_chunks = [
        SourceChunk(
            text=r["text"],
            score=round(r["score"], 4)
        )
        for r in results
    ]

    return AnswerResponse(
        question=request.question,
        answer=best_answer,
        source_chunks=source_chunks
    )