import time
import asyncio
from http import HTTPStatus
from typing import Dict, List
from pathlib import Path

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
from constants import SUPPORTED_EXTENSIONS

router = APIRouter()

batch_store = {}


def check_embedding_service(
    embedding_service: EmbeddingService | None, endpoint: str
) -> EmbeddingService:
    if embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized",
        )
    return embedding_service


@router.post("/api/v1/batch/create")
async def create_batch():
    batch_id = f"batch_{int(time.time() * 1000)}"
    batch_store[batch_id] = {
        "collected_files": [],
        "extracted_texts": {},
        "chunked_texts": {},
        "embeddings": [],
        "file_map": {},
        "status": {
            "collection": {"current": 0, "total": 0, "complete": False},
            "extraction": {"current": 0, "total": 0, "complete": False},
            "chunking": {"current": 0, "total": 0, "complete": False},
            "embedding": {"current": 0, "total": 0, "complete": False},
        }
    }
    return JSONResponse(content={"batch_id": batch_id})


@router.post("/api/v1/batch/{batch_id}/collect_files")
async def collect_batch_files(
    batch_id: str,
    request: dict
):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = batch_store[batch_id]
    mode = request.get("mode")
    paths = request.get("paths", [])
    include_subdirs = request.get("include_subdirs", False)
    
    collected_files = []
    
    try:
        if mode == "directory" and paths:
            directory = Path(paths[0])
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Invalid directory: {directory}")
            
            if include_subdirs:
                for ext in SUPPORTED_EXTENSIONS:
                    pattern = f"**/*{ext}"
                    for filepath in directory.glob(pattern):
                        if filepath.is_file():
                            collected_files.append(str(filepath))
            else:
                for ext in SUPPORTED_EXTENSIONS:
                    pattern = f"*{ext}"
                    for filepath in directory.glob(pattern):
                        if filepath.is_file():
                            collected_files.append(str(filepath))
        
        elif mode == "files":
            for filepath in paths:
                path = Path(filepath)
                if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                    collected_files.append(str(filepath))
        
        batch["collected_files"] = collected_files
        batch["status"]["collection"]["total"] = len(collected_files)
        batch["status"]["collection"]["current"] = len(collected_files)
        batch["status"]["collection"]["complete"] = True
        
        return JSONResponse(content={
            "status": "success",
            "files_collected": len(collected_files)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File collection failed: {str(e)}")


@router.post("/api/v1/batch/{batch_id}/extract_texts")
async def extract_batch_texts(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = batch_store[batch_id]
    
    if not batch["collected_files"]:
        raise HTTPException(status_code=400, detail="No files collected")
    
    batch["status"]["extraction"]["total"] = len(batch["collected_files"])
    batch["status"]["extraction"]["current"] = 0
    
    def extract_all():
        for idx, filepath in enumerate(batch["collected_files"]):
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                text = extract_text_from_file(content, filepath)
                batch["extracted_texts"][idx] = text
                batch["file_map"][idx] = filepath
                batch["status"]["extraction"]["current"] = idx + 1
            except Exception as e:
                continue
        
        batch["status"]["extraction"]["complete"] = True
    
    await asyncio.get_event_loop().run_in_executor(None, extract_all)
    
    return JSONResponse(content={
        "status": "complete",
        "texts_extracted": len(batch["extracted_texts"]),
        "total_files": len(batch["collected_files"])
    })


@router.post("/api/v1/batch/{batch_id}/chunk_texts")
async def chunk_batch_texts(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    embedding_service = check_embedding_service(app_state.embedding_service, "chunk")
    batch = batch_store[batch_id]
    
    if not batch["extracted_texts"]:
        raise HTTPException(status_code=400, detail="No texts to chunk")
    
    batch["status"]["chunking"]["total"] = len(batch["extracted_texts"])
    batch["status"]["chunking"]["current"] = 0
    
    def chunk_all():
        for doc_id, text in batch["extracted_texts"].items():
            chunks = embedding_service.split_text_into_chunks(text)
            batch["chunked_texts"][doc_id] = chunks
            batch["status"]["chunking"]["current"] = doc_id + 1
        
        batch["status"]["chunking"]["complete"] = True
    
    await asyncio.get_event_loop().run_in_executor(None, chunk_all)
    
    total_chunks = sum(len(chunks) for chunks in batch["chunked_texts"].values())
    
    return JSONResponse(content={
        "status": "complete",
        "documents_chunked": len(batch["chunked_texts"]),
        "total_chunks": total_chunks
    })


@router.post("/api/v1/batch/{batch_id}/embed_chunks")
async def embed_batch_chunks(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    embedding_service = check_embedding_service(app_state.embedding_service, "embed")
    batch = batch_store[batch_id]
    
    if not batch["chunked_texts"]:
        raise HTTPException(status_code=400, detail="No chunks to embed")
    
    all_chunks = []
    chunk_to_doc = []
    
    for doc_id, chunks in batch["chunked_texts"].items():
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_to_doc.append((doc_id, chunk_idx))
    
    batch["status"]["embedding"]["total"] = len(all_chunks)
    batch["status"]["embedding"]["current"] = 0
    
    embeddings_array = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: embedding_service.gpu_model.encode_document(
            all_chunks,
            batch_size=runtime_settings.processing_batch_size,
            normalize_embeddings=True
        )
    )
    
    batch["embeddings"] = []
    embeddings_by_doc = {}
    
    for idx, (doc_id, chunk_idx) in enumerate(chunk_to_doc):
        if doc_id not in embeddings_by_doc:
            embeddings_by_doc[doc_id] = []
        
        embeddings_by_doc[doc_id].append({
            "chunk_number": chunk_idx + 1,
            "chunk": all_chunks[idx],
            "embedding": embeddings_array[idx].tolist()
        })
        batch["status"]["embedding"]["current"] = idx + 1
    
    for doc_id, doc_embeddings in embeddings_by_doc.items():
        batch["embeddings"].append({
            "id": doc_id,
            "embeddings": doc_embeddings
        })
    
    batch["status"]["embedding"]["complete"] = True
    
    embedding_service.cleanup_gpu_memory()
    
    return JSONResponse(content={
        "status": "complete",
        "documents_embedded": len(batch["embeddings"]),
        "total_embeddings": len(all_chunks)
    })


@router.get("/api/v1/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = batch_store[batch_id]
    
    return JSONResponse(content={
        "status": batch["status"],
        "files_collected": len(batch["collected_files"]),
        "texts_extracted": len(batch["extracted_texts"]),
        "documents_chunked": len(batch["chunked_texts"]),
        "documents_embedded": len(batch["embeddings"])
    })


@router.post("/api/v1/batch/{batch_id}/query")
async def query_batch(batch_id: str, request: QueryRequest):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    embedding_service = check_embedding_service(app_state.embedding_service, "query")
    batch = batch_store[batch_id]
    
    if not batch["embeddings"]:
        raise HTTPException(status_code=400, detail="No embeddings available")
    
    validate_text_length(request.text, "query")
    query_embedding = await embedding_service.generate_query_embedding(request.text)
    
    def compute_similarities():
        import numpy as np
        qemb = np.array(query_embedding)
        
        similarities = []
        for doc_data in batch["embeddings"]:
            doc_id = doc_data["id"]
            filename = Path(batch["file_map"][doc_id]).name
            
            for emb_data in doc_data["embeddings"]:
                chunk_vec = np.array(emb_data["embedding"])
                sim = np.dot(qemb, chunk_vec) / (np.linalg.norm(qemb) * np.linalg.norm(chunk_vec))
                similarities.append({
                    "document": filename,
                    "document_id": doc_id,
                    "chunk_number": emb_data["chunk_number"],
                    "similarity": float(sim),
                    "chunk": emb_data["chunk"]
                })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:10]
    
    top_results = await asyncio.get_event_loop().run_in_executor(None, compute_similarities)
    
    return JSONResponse(content={"results": top_results})


@router.get("/api/v1/batch/{batch_id}/summary")
async def get_batch_summary(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = batch_store[batch_id]
    
    summary = []
    for doc_data in batch["embeddings"]:
        doc_id = doc_data["id"]
        filename = Path(batch["file_map"][doc_id]).name
        embeddings = doc_data["embeddings"]
        
        total_chars = sum(len(e["chunk"]) for e in embeddings)
        avg_size = int(total_chars / len(embeddings)) if embeddings else 0
        
        summary.append({
            "filename": filename,
            "chunks": len(embeddings),
            "total_chars": total_chars,
            "avg_chunk_size": avg_size
        })
    
    return JSONResponse(content={"summary": summary})


@router.delete("/api/v1/batch/{batch_id}")
async def delete_batch(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    del batch_store[batch_id]
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