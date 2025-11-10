import asyncio
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BitsAndBytesConfig

import app_state
from config import runtime_settings
from embedding_service import EmbeddingService
from utils import logger


async def load_embedding_service(custom_settings=None):
    max_retries = 3
    retry_delay = 5

    settings_to_use = custom_settings if custom_settings else runtime_settings

    if app_state.embedding_service is not None:
        logger.info("Cleaning up existing embedding service...")
        try:
            if hasattr(app_state.embedding_service, 'cleanup_gpu_memory'):
                app_state.embedding_service.cleanup_gpu_memory()
            del app_state.embedding_service.model
            del app_state.embedding_service.gpu_model
            del app_state.embedding_service
            app_state.embedding_service = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Old model cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Attempting to initialize embedding service (attempt {attempt + 1}/{max_retries})"
            )

            model_name = settings_to_use.transformer_model_name
            if hasattr(model_name, 'value'):
                model_name = model_name.value

            logger.info(f"Loading model: {model_name}")

            is_qwen3 = "Qwen3-Embedding" in model_name or "Qwen/Qwen3" in model_name
            
            tokenizer_kwargs = {
                "model_max_length": 8192
            }
            
            if is_qwen3:
                tokenizer_kwargs["padding_side"] = "left"
                logger.info("Using left padding for Qwen3 model")
            else:
                logger.info("Using default padding (right) for non-Qwen3 model")

            model_kwargs = {
                "attn_implementation": "sdpa",
            }
            
            if settings_to_use.use_quantization and not settings_to_use.force_cpu:
                if not torch.cuda.is_available():
                    logger.warning("Quantization requested but CUDA not available. Loading without quantization.")
                    model_kwargs["torch_dtype"] = torch.float32
                else:
                    logger.info("Configuring 4-bit quantization with bitsandbytes")
                    
                    compute_dtype_map = {
                        "bfloat16": torch.bfloat16,
                        "float16": torch.float16,
                        "float32": torch.float32,
                    }
                    compute_dtype = compute_dtype_map.get(
                        settings_to_use.compute_dtype, 
                        torch.bfloat16
                    )
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=settings_to_use.quantization_type,
                        bnb_4bit_use_double_quant=settings_to_use.use_double_quant,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    
                    logger.info(
                        f"Quantization config: type={settings_to_use.quantization_type}, "
                        f"double_quant={settings_to_use.use_double_quant}, "
                        f"compute_dtype={settings_to_use.compute_dtype}"
                    )
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model = SentenceTransformer(
                model_name,
                revision=settings_to_use.transformer_model_version,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs
            )

            if settings_to_use.use_quantization and torch.cuda.is_available() and not settings_to_use.force_cpu:
                is_quantized = False
                try:
                    if hasattr(model[0], 'auto_model'):
                        base_model = model[0].auto_model
                        is_quantized = hasattr(base_model, 'hf_quantizer') and base_model.hf_quantizer is not None
                        
                        if is_quantized:
                            logger.info("✓ Model is quantized with bitsandbytes")
                            
                            for name, param in base_model.named_parameters():
                                if 'weight' in name:
                                    logger.info(f"Sample weight dtype: {param.dtype}, device: {param.device}")
                                    break
                        else:
                            logger.warning("✗ Model is NOT quantized - quantization may have failed")
                except Exception as e:
                    logger.warning(f"Could not verify quantization status: {str(e)}")

                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")

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

            logger.info(
                f"Embedding service initialized successfully with model: {model_name}"
                + (" (quantized)" if settings_to_use.use_quantization and torch.cuda.is_available() and not settings_to_use.force_cpu else "")
            )
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