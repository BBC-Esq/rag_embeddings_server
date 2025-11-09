import platform
import sys
import os
import shutil
from pathlib import Path
import logging
from http import HTTPStatus

from fastapi import HTTPException
from torch.cuda import OutOfMemoryError

from config import runtime_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def extract_text_from_file(content: bytes, filename: str) -> str:
    try:
        text = content.decode('utf-8')
        
        if filename.lower().endswith('.html'):
            import html2text
            h = html2text.HTML2Text()
            h.unicode_snob = True
            h.body_width = 0
            h.skip_internal_links = True
            h.ignore_anchors = True
            h.ignore_images = True
            h.ignore_emphasis = True
            h.ignore_links = True
            h.single_line_break = True
            h.mark_code = False
            h.decode_errors = 'ignore'
            h.bypass_tables = False
            text = h.handle(text)
        
        return text
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode file {filename}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to extract text from {filename}: {str(e)}")


def validate_text_length(
    text: str, endpoint: str, doc_id: int | None = None
) -> None:
    text_length = len(text.strip())
    if text_length < runtime_settings.min_text_length:
        error_msg = f"Text length ({text_length}) below minimum ({runtime_settings.min_text_length})"
        if doc_id is not None:
            error_msg = f"Document {doc_id}: {error_msg}"
        raise ValueError(error_msg)

    max_length = (
        runtime_settings.max_query_length
        if endpoint == "query"
        else runtime_settings.max_text_length
    )
    label = "Query" if endpoint == "query" else "Text"
    if text_length > max_length:
        raise ValueError(
            f"{label} length ({text_length}) exceeds maximum ({max_length})"
        )


def handle_exception(e: Exception, endpoint: str) -> None:
    match e:
        case UnicodeDecodeError():
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Invalid UTF-8 encoding in text",
            )
        case ValueError():
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e)
            )
        case OutOfMemoryError():
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="GPU memory exhausted",
            )
        case _:
            raise e