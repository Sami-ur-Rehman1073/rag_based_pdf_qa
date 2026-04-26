---
title: RAG PDF Q&A System
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 📄 RAG-Based PDF Question Answering System

A fully local Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask questions about it — no paid APIs required.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | sentence-transformers |
| Vector DB | FAISS |
| PDF Parsing | PyPDF |
| Frontend | HTML + Tailwind CSS |
| Deployment | HuggingFace Spaces (Docker) |

## 🚀 Features

- Upload any PDF
- Automatically extract and chunk text
- Generate local embeddings
- Store vectors in FAISS
- Ask natural language questions
- Get answers with source text

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Sami-ur-Rehman1073/rag_based_pdf_qa.git
cd rag_based_pdf_qa
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Run the backend
```bash
uvicorn main:app --reload
```

### 5. Open the app
Go to `http://127.0.0.1:8000`

## 📁 Project Structure