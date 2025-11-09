# route_monitoring.py
import torch
from fastapi import APIRouter, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

import app_state
from config import runtime_settings, EmbeddingModel
from service_loader import load_embedding_service
from utils import logger

router = APIRouter()


class ReloadSettings(BaseModel):
    transformer_model_name: str
    transformer_model_version: str
    use_query_prompt: bool
    query_prompt_name: str
    use_document_prompt: bool
    document_prompt_name: str | None
    max_tokens: int
    overlap_ratio: float
    min_text_length: int
    max_query_length: int
    max_text_length: int
    max_batch_size: int
    processing_batch_size: int
    max_workers: int
    force_cpu: bool
    enable_metrics: bool


@router.get("/")
async def heartbeat():
    return "Heartbeat detected."


@router.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    
    model_name = runtime_settings.transformer_model_name
    if hasattr(model_name, 'value'):
        model_name = model_name.value
    
    model_info = {
        "name": model_name,
        "version": runtime_settings.transformer_model_version,
    }
    
    if app_state.embedding_service:
        model_info["supports_prompts"] = app_state.embedding_service.supports_prompts
        if app_state.embedding_service.supports_prompts:
            model_info["available_prompts"] = list(app_state.embedding_service.model.prompts.keys())
    
    return {
        "status": (
            "healthy" if app_state.embedding_service else "service_unavailable"
        ),
        "model_loaded": app_state.embedding_service is not None,
        "model_info": model_info,
        "gpu_available": gpu_available and not runtime_settings.force_cpu,
        "prompt_config": {
            "use_query_prompt": runtime_settings.use_query_prompt,
            "query_prompt_name": runtime_settings.query_prompt_name,
            "use_document_prompt": runtime_settings.use_document_prompt,
            "document_prompt_name": runtime_settings.document_prompt_name,
        }
    }


@router.post("/reload_service")
async def reload_service(new_settings: ReloadSettings):
    try:
        logger.info("Reloading service with new settings")
        
        if app_state.embedding_service:
            logger.info("Cleaning up existing service...")
            app_state.embedding_service.cleanup_gpu_memory()
            app_state.embedding_service = None
        
        class TempSettings:
            pass
        
        temp = TempSettings()
        for key, value in new_settings.model_dump().items():
            if key == 'transformer_model_name':
                for model_enum in EmbeddingModel:
                    if model_enum.value == value:
                        value = model_enum
                        break
            setattr(temp, key, value)
        
        success = await load_embedding_service(temp)
        
        if success:
            logger.info("Service reloaded successfully with new settings")
            return {
                "status": "success",
                "message": "Service reloaded successfully"
            }
        else:
            logger.error("Service reload failed")
            return {
                "status": "error",
                "message": "Failed to reload service"
            }
    except Exception as e:
        logger.error(f"Error during service reload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service reload failed: {str(e)}")


@router.get("/current_settings")
async def get_current_settings():
    model_name = runtime_settings.transformer_model_name
    if hasattr(model_name, 'value'):
        model_name = model_name.value
    
    return {
        "transformer_model_name": model_name,
        "transformer_model_version": runtime_settings.transformer_model_version,
        "use_query_prompt": runtime_settings.use_query_prompt,
        "query_prompt_name": runtime_settings.query_prompt_name,
        "use_document_prompt": runtime_settings.use_document_prompt,
        "document_prompt_name": runtime_settings.document_prompt_name,
        "max_tokens": runtime_settings.max_tokens,
        "overlap_ratio": runtime_settings.overlap_ratio,
        "min_text_length": runtime_settings.min_text_length,
        "max_query_length": runtime_settings.max_query_length,
        "max_text_length": runtime_settings.max_text_length,
        "max_batch_size": runtime_settings.max_batch_size,
        "processing_batch_size": runtime_settings.processing_batch_size,
        "max_workers": runtime_settings.max_workers,
        "force_cpu": runtime_settings.force_cpu,
        "enable_metrics": runtime_settings.enable_metrics,
    }


@router.get("/metrics")
async def metrics():
    if not runtime_settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)