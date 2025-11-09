from enum import Enum
import re


class TextCleaningMode(str, Enum):
    ASCII_ONLY = "ascii_only"
    ASCII_EXTENDED = "ascii_extended"
    UNICODE_SAFE = "unicode_safe"
    WHITESPACE_ONLY = "whitespace_only"
    NO_CLEANING = "no_cleaning"


def clean_ascii_only(text: str) -> str:
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


def clean_ascii_extended(text: str) -> str:
    if not text:
        return ""

    allowed_ranges = [
        (32, 127),
        (160, 255),
        (0x2010, 0x2027),
        (0x2030, 0x205E),
        (0x20A0, 0x20CF),
    ]

    def is_allowed(char):
        if char in "\n\t":
            return True
        code = ord(char)
        return any(start <= code <= end for start, end in allowed_ranges)

    text = "".join(char for char in text if is_allowed(char))

    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def clean_unicode_safe(text: str) -> str:
    if not text:
        return ""

    def is_allowed(char):
        if char in "\n\t":
            return True
        code = ord(char)
        if code < 32:
            return False
        if 0xD800 <= code <= 0xDFFF:
            return False
        if code in (0xFFFE, 0xFFFF):
            return False
        return True

    text = "".join(char for char in text if is_allowed(char))

    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def clean_whitespace_only(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    return text


def clean_no_cleaning(text: str) -> str:
    return text if text else ""


def preprocess_text(text: str, mode: TextCleaningMode) -> str:
    cleaners = {
        TextCleaningMode.ASCII_ONLY: clean_ascii_only,
        TextCleaningMode.ASCII_EXTENDED: clean_ascii_extended,
        TextCleaningMode.UNICODE_SAFE: clean_unicode_safe,
        TextCleaningMode.WHITESPACE_ONLY: clean_whitespace_only,
        TextCleaningMode.NO_CLEANING: clean_no_cleaning,
    }

    cleaner = cleaners.get(mode, clean_ascii_only)
    cleaned_text = cleaner(text)

    if not cleaned_text:
        raise ValueError("Text is empty after cleaning.")

    return cleaned_text