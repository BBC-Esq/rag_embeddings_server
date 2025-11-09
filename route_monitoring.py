import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app_state
from config import runtime_settings, EmbeddingModel
from service_loader import load_embedding_service
from text_cleaning import TextCleaningMode
from utils import logger

router = APIRouter()


class ReloadSettings(BaseModel):
    transformer_model_name: str
    transformer_model_version: str
    chunk_size: int
    chunk_overlap: int
    min_text_length: int
    max_query_length: int
    max_text_length: int
    max_batch_size: int
    processing_batch_size: int
    max_workers: int
    force_cpu: bool
    text_cleaning_mode: str


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
            elif key == 'text_cleaning_mode':
                for mode_enum in TextCleaningMode:
                    if mode_enum.value == value:
                        value = mode_enum
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

    cleaning_mode = runtime_settings.text_cleaning_mode
    if hasattr(cleaning_mode, 'value'):
        cleaning_mode = cleaning_mode.value

    return {
        "transformer_model_name": model_name,
        "transformer_model_version": runtime_settings.transformer_model_version,
        "chunk_size": runtime_settings.chunk_size,
        "chunk_overlap": runtime_settings.chunk_overlap,
        "min_text_length": runtime_settings.min_text_length,
        "max_query_length": runtime_settings.max_query_length,
        "max_text_length": runtime_settings.max_text_length,
        "max_batch_size": runtime_settings.max_batch_size,
        "processing_batch_size": runtime_settings.processing_batch_size,
        "max_workers": runtime_settings.max_workers,
        "force_cpu": runtime_settings.force_cpu,
        "text_cleaning_mode": cleaning_mode,
    }