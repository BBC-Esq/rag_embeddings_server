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