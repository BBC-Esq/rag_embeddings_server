import asyncio

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import app_state
from config import runtime_settings
from embedding_service import EmbeddingService
from utils import logger


async def load_embedding_service(custom_settings=None):
    max_retries = 3
    retry_delay = 5
    
    settings_to_use = custom_settings if custom_settings else runtime_settings

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Attempting to initialize embedding service (attempt {attempt + 1}/{max_retries})"
            )
            
            model_name = settings_to_use.transformer_model_name
            if hasattr(model_name, 'value'):
                model_name = model_name.value
            
            logger.info(f"Loading model: {model_name}")
            
            model = SentenceTransformer(
                model_name,
                revision=settings_to_use.transformer_model_version,
                model_kwargs={
                    "attn_implementation": "sdpa",
                    "torch_dtype": torch.float32
                },
                tokenizer_kwargs={
                    "padding_side": "left",
                    "model_max_length": 8192
                }
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            app_state.embedding_service = EmbeddingService(
                model=model,
                tokenizer=tokenizer,
                chunk_size=settings_to_use.chunk_size,
                chunk_overlap=settings_to_use.chunk_overlap,
                processing_batch_size=settings_to_use.processing_batch_size,
                max_workers=settings_to_use.max_workers,
            )
            
            if custom_settings:
                runtime_settings.update(**{
                    k: getattr(custom_settings, k) 
                    for k in dir(custom_settings) 
                    if not k.startswith('_')
                })
            
            logger.info(f"Embedding service initialized successfully with model: {model_name}")
            return True
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            )
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Failed to initialize embedding service")
                return False
    return False