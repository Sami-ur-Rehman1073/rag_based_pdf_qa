# backend/services/embedding_service.py

import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------
# Load the embedding model ONCE at module level
#
# Why at module level?
# Loading a model is expensive (takes 2-3 seconds)
# By loading it once here, every function call
# reuses the same loaded model instantly
#
# all-MiniLM-L6-v2:
# - Small and fast (runs on CPU)
# - Produces 384-dimensional vectors
# - Great accuracy for Q&A retrieval tasks
# - Downloads automatically on first run (~80MB)
# -----------------------------------------------
print("Loading embedding model... (this may take a moment on first run)")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")


# -----------------------------------------------
# embed_chunks()
#
# Takes a list of text chunks
# Returns a 2D numpy array of shape:
#   (number_of_chunks, 384)
#
# Example:
#   Input:  ["ML is great", "Python is easy"]
#   Output: array([[0.21, 0.84, ...],   ← vector for chunk 1
#                  [0.11, 0.63, ...]])  ← vector for chunk 2
# -----------------------------------------------
def embed_chunks(chunks: list[str]) -> np.ndarray:

    if not chunks:
        raise ValueError("Cannot embed an empty list of chunks.")

    # convert_to_numpy=True returns a numpy array
    # which is what FAISS expects
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True   # Shows progress in terminal
    )

    return embeddings


# -----------------------------------------------
# embed_question()
#
# Takes a single question string
# Returns a 1D numpy array of shape (384,)
#
# We reshape it to (1, 384) because FAISS
# expects a 2D array even for single queries
# -----------------------------------------------
def embed_question(question: str) -> np.ndarray:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding = model.encode(
        [question],              # Wrap in list → model expects a list
        convert_to_numpy=True
    )

    # embedding shape is (1, 384) — ready for FAISS search
    return embedding