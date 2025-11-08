import torch
from fastapi import APIRouter, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import app_state  # ← Change from 'import main'
from config import settings

router = APIRouter()


@router.get("/")
async def heartbeat():
    """Simple heartbeat endpoint"""
    return "Heartbeat detected."


@router.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    gpu_available = torch.cuda.is_available()
    return {
        "status": (
            "healthy" if app_state.embedding_service else "service_unavailable"  # ← Change
        ),
        "model_loaded": app_state.embedding_service is not None,  # ← Change
        "gpu_available": gpu_available and not settings.force_cpu,
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)