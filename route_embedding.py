import time
import asyncio
from http import HTTPStatus
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

import app_state
from config import runtime_settings
from embedding_service import EmbeddingService
from schemas import (
    QueryRequest,
    QueryResponse,
)
from text_cleaning import preprocess_text
from utils import (
    handle_exception,
    validate_text_length,
    extract_text_from_file,
)

router = APIRouter()

session_store = {}


def check_embedding_service(
    embedding_service: EmbeddingService | None, endpoint: str
) -> EmbeddingService:
    if embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized",
        )
    return embedding_service


@router.post("/api/v1/session/create")
async def create_session():
    session_id = f"session_{int(time.time() * 1000)}"
    session_store[session_id] = {
        "extracted_texts": {},
        "chunked_texts": {},
        "embeddings": {},
        "file_map": {},
        "chunk_progress": {"current": 0, "total": 0},
        "embed_progress": {"current": 0, "total": 0},
    }
    return JSONResponse(content={"session_id": session_id})


@router.post("/api/v1/session/{session_id}/upload")
async def upload_files_to_session(session_id: str, files: list[UploadFile] = File(...)):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_store[session_id]
    results = []

    for idx, file in enumerate(files):
        try:
            content = await file.read()
            text = extract_text_from_file(content, file.filename)

            file_id = len(session["extracted_texts"])
            session["extracted_texts"][file_id] = text
            session["file_map"][file_id] = file.filename

            results.append({
                "id": file_id,
                "filename": file.filename,
                "text": text,
                "success": True,
                "size": len(text),
                "error": None
            })
        except Exception as e:
            results.append({
                "id": -1,
                "filename": file.filename,
                "text": "",
                "success": False,
                "size": 0,
                "error": str(e)
            })

    return JSONResponse(content={"files": results})


@router.post("/api/v1/session/{session_id}/chunk")
async def chunk_session_texts(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    embedding_service = check_embedding_service(app_state.embedding_service, "chunk")
    session = session_store[session_id]

    if "extracted_texts" not in session or not session["extracted_texts"]:
        raise HTTPException(status_code=400, detail="No texts to chunk. Please re-upload files.")

    session["chunk_progress"] = {"current": 0, "total": len(session["extracted_texts"])}

    def chunk_all():
        for doc_id, text in session["extracted_texts"].items():
            chunks = embedding_service.split_text_into_chunks(text)
            session["chunked_texts"][doc_id] = chunks
            session["chunk_progress"]["current"] += 1

    await asyncio.get_event_loop().run_in_executor(None, chunk_all)

    total_chunks = sum(len(chunks) for chunks in session["chunked_texts"].values())

    return JSONResponse(content={
        "status": "complete",
        "documents_chunked": len(session["chunked_texts"]),
        "total_chunks": total_chunks
    })


@router.post("/api/v1/session/{session_id}/embed")
async def embed_session_chunks(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    embedding_service = check_embedding_service(app_state.embedding_service, "embed")
    session = session_store[session_id]

    if not session["chunked_texts"]:
        raise HTTPException(status_code=400, detail="No chunks to embed")

    all_chunks = []
    chunk_to_doc = []

    for doc_id, chunks in session["chunked_texts"].items():
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_to_doc.append((doc_id, chunk_idx))

    session["embed_progress"] = {"current": 0, "total": len(all_chunks)}

    embeddings_array = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: embedding_service.gpu_model.encode_document(
            all_chunks,
            batch_size=runtime_settings.processing_batch_size,
            normalize_embeddings=True
        )
    )
    
    for idx, (doc_id, chunk_idx) in enumerate(chunk_to_doc):
        if doc_id not in session["embeddings"]:
            session["embeddings"][doc_id] = []
        session["embeddings"][doc_id].append({
            "chunk_number": chunk_idx + 1,
            "chunk": all_chunks[idx],
            "embedding": embeddings_array[idx].tolist()
        })

    embedding_service.cleanup_gpu_memory()

    # del session["extracted_texts"] # uncomment to delete, but will prevent re-embedding with a different model without resending

    return JSONResponse(content={
        "status": "complete",
        "documents_embedded": len(session["embeddings"]),
        "total_embeddings": len(all_chunks)
    })


@router.get("/api/v1/session/{session_id}/progress")
async def get_session_progress(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_store[session_id]
    return JSONResponse(content={
        "chunk_progress": session["chunk_progress"],
        "embed_progress": session["embed_progress"]
    })


@router.post("/api/v1/session/{session_id}/query")
async def query_session(session_id: str, request: QueryRequest):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    embedding_service = check_embedding_service(app_state.embedding_service, "query")
    session = session_store[session_id]

    if not session["embeddings"]:
        raise HTTPException(status_code=400, detail="No embeddings available")

    validate_text_length(request.text, "query")
    query_embedding = await embedding_service.generate_query_embedding(request.text)

    import numpy as np
    qemb = np.array(query_embedding)

    similarities = []
    for doc_id, doc_embeddings in session["embeddings"].items():
        for emb_data in doc_embeddings:
            chunk_vec = np.array(emb_data["embedding"])
            sim = np.dot(qemb, chunk_vec) / (np.linalg.norm(qemb) * np.linalg.norm(chunk_vec))
            similarities.append({
                "document": session["file_map"][doc_id],
                "document_id": doc_id,
                "chunk_number": emb_data["chunk_number"],
                "similarity": float(sim),
                "chunk": emb_data["chunk"]
            })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:10]

    return JSONResponse(content={"results": top_results})


@router.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    del session_store[session_id]
    return JSONResponse(content={"status": "deleted"})


@router.post("/api/v1/embed/query", response_model=QueryResponse)
async def create_query_embedding(request: QueryRequest):
    embedding_service = check_embedding_service(app_state.embedding_service, "query")
    try:
        validate_text_length(request.text, "query")
        embedding = await embedding_service.generate_query_embedding(request.text)
        return QueryResponse(embedding=embedding)
    except Exception as e:
        handle_exception(e, "query")


@router.post("/api/v1/embed/text")
async def create_text_embedding(request: Request):
    embedding_service = check_embedding_service(app_state.embedding_service, "text")
    try:
        raw_text = await request.body()
        text = raw_text.decode("utf-8")
        validate_text_length(text, "text")
        result = await embedding_service.generate_text_embeddings({0: text})

        text_length = len(text.strip())
        if text_length > runtime_settings.chunk_size * 10:
            embedding_service.cleanup_gpu_memory()

        return result[0]
    except Exception as e:
        handle_exception(e, "text")


@router.post("/api/v1/validate/text")
async def validate_text(request: dict):
    try:
        text = request.get("text", "")
        doc_id = request.get("id", 0)
        processed_text = preprocess_text(text, runtime_settings.text_cleaning_mode)
        return {
            "id": doc_id,
            "original_text": text,
            "processed_text": processed_text,
            "is_valid": True,
        }
    except Exception as e:
        return {
            "id": request.get("id", 0),
            "original_text": request.get("text", ""),
            "error": str(e),
            "is_valid": False,
        }