from pydantic import BaseModel

# -----------------------------------------------
# Request model for /ask-question
# -----------------------------------------------
class QuestionRequest(BaseModel):
    question: str

# -----------------------------------------------
# A single retrieved chunk with its distance score
#
# distance: L2 distance from FAISS
#   lower  = more similar = better match
#   higher = less similar = weaker match
# -----------------------------------------------
class SourceChunk(BaseModel):
    text: str
    score: float

# -----------------------------------------------
# Response model for /ask-question
#
# answer       : the most relevant chunk's text
#                (top result = best answer)
# source_chunks: all retrieved chunks with scores
#                so user can see context
# -----------------------------------------------
class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_chunks: list[SourceChunk]

# -----------------------------------------------
# Generic message response (used for /upload-pdf)
# -----------------------------------------------
class MessageResponse(BaseModel):
    message: str
    filename: str | None = None
    chunks_created: int | None = None