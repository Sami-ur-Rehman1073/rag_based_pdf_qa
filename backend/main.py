import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.pdf_utils import extract_text_from_pdf
from utils.text_utils import split_text_into_chunks
from services.embedding_service import embed_chunks
from models.schemas import MessageResponse

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
# Health Check
# -----------------------------------------------
@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "status": "running",
        "message": "RAG PDF Q&A backend is live!",
        "endpoints": {
            "upload_pdf": "POST /upload-pdf",
            "ask_question": "POST /ask-question",
            "docs": "GET /docs"
        }
    }

# -----------------------------------------------
# POST /upload-pdf
#
# 1. Validate file is PDF
# 2. Save to disk
# 3. Extract text
# 4. Split into chunks
# 5. Embed all chunks        ← NEW THIS STEP
# 6. Return success
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

    # ---- Embed all chunks ----
    # This converts every chunk into a vector
    # We print shape in terminal for debugging
    try:
        embeddings = embed_chunks(chunks)
        print(f"Embeddings shape: {embeddings.shape}")
        # e.g. (42, 384) means 42 chunks, each with 384 numbers
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )

    # ---- Return success ----
    return MessageResponse(
        message=f"PDF processed and embedded successfully. {len(chunks)} chunks created.",
        filename=file.filename,
        chunks_created=len(chunks)
    )