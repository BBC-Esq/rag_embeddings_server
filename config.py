# config.py
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingModel(str, Enum):
    MODERNBERT = "freelawproject/modernbert-embed-base_finetune_512"
    QWEN3 = "Qwen/Qwen3-Embedding-0.6B"


class Settings(BaseSettings):
    transformer_model_name: EmbeddingModel = Field(
        EmbeddingModel.MODERNBERT,
        description="Name of the transformer model to use",
    )
    transformer_model_version: str = Field(
        "main",
        description="Version of the transformer model to use",
    )
    use_query_prompt: bool = Field(
        True,
        description="Whether to use a prompt for query embeddings",
    )
    query_prompt_name: str = Field(
        "query",
        description="Name of the prompt to use for query embeddings (if model supports it)",
    )
    use_document_prompt: bool = Field(
        False,
        description="Whether to use a prompt for document embeddings",
    )
    document_prompt_name: str | None = Field(
        None,
        description="Name of the prompt to use for document embeddings (if model supports it)",
    )
    max_tokens: int = Field(
        512, ge=256, le=10000, description="Maximum tokens per chunk"
    )
    overlap_ratio: float = Field(
        0.004,
        ge=0,
        le=0.01,
        description="Ratio to calculate number of sentence overlap between chunks",
    )
    min_text_length: int = 1
    max_query_length: int = 1000
    max_text_length: int = 10_000_000
    max_batch_size: int = 100
    processing_batch_size: int = 8
    max_workers: int = 4
    pool_timeout: int = 3600
    force_cpu: bool = False
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


class RuntimeSettings:
    def __init__(self):
        self.transformer_model_name = settings.transformer_model_name
        self.transformer_model_version = settings.transformer_model_version
        self.use_query_prompt = settings.use_query_prompt
        self.query_prompt_name = settings.query_prompt_name
        self.use_document_prompt = settings.use_document_prompt
        self.document_prompt_name = settings.document_prompt_name
        self.max_tokens = settings.max_tokens
        self.overlap_ratio = settings.overlap_ratio
        self.min_text_length = settings.min_text_length
        self.max_query_length = settings.max_query_length
        self.max_text_length = settings.max_text_length
        self.max_batch_size = settings.max_batch_size
        self.processing_batch_size = settings.processing_batch_size
        self.max_workers = settings.max_workers
        self.pool_timeout = settings.pool_timeout
        self.force_cpu = settings.force_cpu
        self.enable_metrics = settings.enable_metrics
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


runtime_settings = RuntimeSettings()