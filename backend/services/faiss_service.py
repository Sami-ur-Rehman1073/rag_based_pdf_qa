# backend/services/faiss_service.py

import os
import pickle
import numpy as np
import faiss

# -----------------------------------------------
# Paths where FAISS index and chunks are stored
# -----------------------------------------------
FAISS_INDEX_PATH = "../storage/faiss_index.index"
CHUNKS_PATH      = "../storage/chunks.pkl"


# -----------------------------------------------
# save_faiss_index()
# Builds and saves FAISS index + chunks to disk
# -----------------------------------------------
def save_faiss_index(
    embeddings: np.ndarray,
    chunks: list[str]
) -> None:

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings cannot be empty.")

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks."
        )

    dimension = embeddings.shape[1]

    # Build FAISS flat L2 index
    index = faiss.IndexFlatL2(dimension)
    embeddings_float32 = np.array(embeddings, dtype=np.float32)
    index.add(embeddings_float32)

    print(f"FAISS index built: {index.ntotal} vectors, dimension {dimension}")

    # Save index to disk
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved: {FAISS_INDEX_PATH}")

    # Save chunks to disk
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved: {CHUNKS_PATH}")


# -----------------------------------------------
# load_faiss_index()
# Loads FAISS index and chunks from disk
# Returns (index, chunks) tuple
# -----------------------------------------------
def load_faiss_index():

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Please upload a PDF first."
        )

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Chunks file not found. Please upload a PDF first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"FAISS index loaded: {index.ntotal} vectors")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"Chunks loaded: {len(chunks)} chunks")

    return index, chunks


# -----------------------------------------------
# search_faiss_index()
#
# Searches FAISS for top_k closest chunks
# Returns list of dicts with text + score
#
# Each result:
#   {
#     "text"  : the original chunk string,
#     "score" : L2 distance (lower = better)
#   }
# -----------------------------------------------
def search_faiss_index(
    question_embedding: np.ndarray,
    top_k: int = 3
) -> list[dict]:

    index, chunks = load_faiss_index()

    # Never ask for more results than we have
    top_k = min(top_k, index.ntotal)

    # FAISS requires float32
    question_embedding = np.array(
        question_embedding, dtype=np.float32
    )

    # Search the index
    # D = distances (lower is better)
    # I = indices of nearest vectors
    D, I = index.search(question_embedding, top_k)

    print(f"Search complete. Top indices: {I[0]}, Distances: {D[0]}")

    # Build results list
    results = []
    for rank, (idx, distance) in enumerate(zip(I[0], D[0])):
        if idx == -1:
            # FAISS returns -1 when no result found
            continue
        results.append({
            "text" : chunks[idx],
            "score": float(distance)   # convert numpy float to Python float
        })

    return results