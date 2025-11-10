from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings
from text_cleaning import TextCleaningMode


class EmbeddingModel(str, Enum):
    FREELAW_MODERNBERT_512 = "freelawproject/modernbert-embed-base_finetune_512"
    FREELAW_MODERNBERT_8192 = "freelawproject/modernbert-embed-base_finetune_8192"
    QWEN3_SMALL = "Qwen/Qwen3-Embedding-0.6B"
    QWEN3_MEDIUM = "Qwen/Qwen3-Embedding-4B"
    QWEN3_LARGE = "Qwen/Qwen3-Embedding-8B"


class Settings(BaseSettings):
    transformer_model_name: EmbeddingModel = Field(
        EmbeddingModel.QWEN3_SMALL,
        description="Name of the transformer model to use",
    )
    transformer_model_version: str = Field(
        "main",
        description="Version of the transformer model to use",
    )
    chunk_size: int = Field(
        1000, ge=100, le=100000, description="Chunk size in characters"
    )
    chunk_overlap: int = Field(
        250, ge=0, le=10000, description="Chunk overlap in characters"
    )
    min_text_length: int = 10
    processing_batch_size: int = 12
    max_workers: int = 4
    pool_timeout: int = 3600
    force_cpu: bool = False
    
    use_quantization: bool = Field(
        True,
        description="Enable 4-bit quantization using bitsandbytes"
    )
    quantization_type: str = Field(
        "nf4",
        description="Quantization type: 'nf4' or 'fp4'"
    )
    use_double_quant: bool = Field(
        True,
        description="Enable nested quantization for additional memory savings"
    )
    compute_dtype: str = Field(
        "bfloat16",
        description="Compute dtype: 'bfloat16', 'float16', or 'float32'"
    )
    
    text_cleaning_mode: TextCleaningMode = Field(
        TextCleaningMode.UNICODE_SAFE,
        description="Text cleaning mode"
    )

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
        self.processing_batch_size = settings.processing_batch_size
        self.max_workers = settings.max_workers
        self.pool_timeout = settings.pool_timeout
        self.force_cpu = settings.force_cpu
        self.use_quantization = settings.use_quantization
        self.quantization_type = settings.quantization_type
        self.use_double_quant = settings.use_double_quant
        self.compute_dtype = settings.compute_dtype
        self.text_cleaning_mode = settings.text_cleaning_mode

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


runtime_settings = RuntimeSettings()