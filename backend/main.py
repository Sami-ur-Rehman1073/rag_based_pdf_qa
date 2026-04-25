from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -----------------------------------------------
# Create the FastAPI application instance
# -----------------------------------------------
app = FastAPI(
    title="RAG PDF Q&A System",
    description="Upload a PDF and ask questions about it — fully local, no paid APIs.",
    version="1.0.0"
)

# -----------------------------------------------
# CORS Middleware
# This allows our HTML frontend (opened directly
# in browser) to talk to this backend without
# being blocked by browser security policies.
# -----------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (fine for local dev)
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, etc.
    allow_headers=["*"],
)

# -----------------------------------------------
# Health Check Route
# Visit http://127.0.0.1:8000/ to confirm
# the server is running correctly
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