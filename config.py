from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingModel(str, Enum):
    MODERNBERT = "freelawproject/modernbert-embed-base_finetune_512"
    QWEN3_small = "Qwen/Qwen3-Embedding-0.6B"
    QWEN_medium = "Qwen/Qwen3-Embedding-4B"


class Settings(BaseSettings):
    transformer_model_name: EmbeddingModel = Field(
        EmbeddingModel.MODERNBERT,
        description="Name of the transformer model to use",
    )
    transformer_model_version: str = Field(
        "main",
        description="Version of the transformer model to use",
    )
    chunk_size: int = Field(
        2048, ge=100, le=100000, description="Chunk size in characters"
    )
    chunk_overlap: int = Field(
        200, ge=0, le=10000, description="Chunk overlap in characters"
    )
    min_text_length: int = 1
    max_query_length: int = 1000
    max_text_length: int = 10_000_000
    max_batch_size: int = 100
    processing_batch_size: int = 8
    max_workers: int = 4
    pool_timeout: int = 3600
    force_cpu: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


class RuntimeSettings:
    def __init__(self):
        self.transformer_model_name = settings.transformer_model_name
        self.transformer_model_version = settings.transformer_model_version
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.min_text_length = settings.min_text_length
        self.max_query_length = settings.max_query_length
        self.max_text_length = settings.max_text_length
        self.max_batch_size = settings.max_batch_size
        self.processing_batch_size = settings.processing_batch_size
        self.max_workers = settings.max_workers
        self.pool_timeout = settings.pool_timeout
        self.force_cpu = settings.force_cpu
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


runtime_settings = RuntimeSettings()