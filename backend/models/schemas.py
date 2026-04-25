

from pydantic import BaseModel

# -----------------------------------------------
# Request model for the /ask-question endpoint
# The user sends a question as a string
# -----------------------------------------------
class QuestionRequest(BaseModel):
    question: str

# -----------------------------------------------
# Response model for the /ask-question endpoint
# We return the answer + the source text chunks
# -----------------------------------------------
class AnswerResponse(BaseModel):
    answer: str
    source_chunks: list[str]

# -----------------------------------------------
# Generic message response (used for /upload-pdf)
# -----------------------------------------------
class MessageResponse(BaseModel):
    message: str
    filename: str | None = None
    chunks_created: int | None = None