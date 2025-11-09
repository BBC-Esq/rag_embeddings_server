import platform
import sys
import os
import shutil
from pathlib import Path
import logging
import re
from http import HTTPStatus

from fastapi import HTTPException
from torch.cuda import OutOfMemoryError

from config import runtime_settings
from metrics import ERROR_COUNT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_cuda_paths():
    if platform.system() != "Windows":
        logger.debug("Skipping CUDA path setup: not on Windows")
        return
    
    venv_base = Path(sys.executable).parent.parent
    nvidia_base = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    
    if not nvidia_base.exists():
        logger.debug("Skipping CUDA path setup: nvidia packages not found")
        return
    
    logger.info("Setting up CUDA paths for Windows")
    cuda_path_runtime = nvidia_base / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base / 'cuda_runtime' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base / 'cublas' / 'bin'
    cudnn_path = nvidia_base / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base / 'cuda_nvcc' / 'bin'
    
    paths_to_add = [
        str(cuda_path_runtime),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
    os.environ['PATH'] = new_value
    
    triton_cuda_path = nvidia_base / 'cuda_runtime'
    current_cuda_path = os.environ.get('CUDA_PATH', '')
    new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
    os.environ['CUDA_PATH'] = new_cuda_path
    
    logger.info("CUDA paths configured successfully")


def clean_triton_cache():
    triton_cache_dir = Path.home() / '.triton'

    if triton_cache_dir.exists():
        try:
            logger.info(f"Removing Triton cache at {triton_cache_dir}")
            shutil.rmtree(triton_cache_dir)
            logger.info("Triton cache successfully removed")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove Triton cache: {e}")
            return False
    else:
        logger.debug("No Triton cache found to clean")
        return True


def clean_text_for_json(text: str) -> str | None:
    if not text:
        return ""

    text = "".join(
        char
        for char in text
        if char == "\n" or char == "\t" or (32 <= ord(char) < 127)
    )

    text = text.replace("\t", " ")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = "\n".join(line.strip() for line in text.split("\n"))

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()
    return text


def preprocess_text(text: str) -> str:
    cleaned_text = clean_text_for_json(text)
    if not cleaned_text:
        raise ValueError("Text is empty after cleaning.")
    return cleaned_text


def validate_text_length(
    text: str, endpoint: str, doc_id: int | None = None
) -> None:
    text_length = len(text.strip())
    if text_length < runtime_settings.min_text_length:
        ERROR_COUNT.labels(
            endpoint=endpoint, error_type="text_too_short"
        ).inc()
        error_msg = f"Text length ({text_length}) below minimum ({runtime_settings.min_text_length})"
        if doc_id is not None:
            error_msg = f"Document {doc_id}: {error_msg}"
        raise ValueError(error_msg)

    max_length = (
        runtime_settings.max_query_length
        if endpoint == "query"
        else runtime_settings.max_text_length
    )
    error_type = "query_too_long" if endpoint == "query" else "text_too_long"
    label = "Query" if endpoint == "query" else "Text"
    if text_length > max_length:
        ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()
        raise ValueError(
            f"{label} length ({text_length}) exceeds maximum ({max_length})"
        )


def handle_exception(e: Exception, endpoint: str) -> None:
    match e:
        case UnicodeDecodeError():
            ERROR_COUNT.labels(
                endpoint=endpoint, error_type="decode_error"
            ).inc()
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Invalid UTF-8 encoding in text",
            )
        case ValueError():
            ERROR_COUNT.labels(
                endpoint=endpoint, error_type="validation_error"
            ).inc()
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e)
            )
        case OutOfMemoryError():
            ERROR_COUNT.labels(endpoint=endpoint, error_type="gpu_error").inc()
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="GPU memory exhausted",
            )
        case _:
            ERROR_COUNT.labels(
                endpoint=endpoint, error_type="processing_error"
            ).inc()
            raise e