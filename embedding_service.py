import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import runtime_settings
from metrics import CHUNK_COUNT, MODEL_LOAD_TIME
from schemas import ChunkEmbedding, TextResponse
from utils import logger, preprocess_text

torch.set_float32_matmul_precision("high")
thread_local = threading.local()


class EmbeddingService:
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: AutoTokenizer,
        chunk_size: int,
        chunk_overlap: int,
        processing_batch_size: int,
        max_workers: int,
    ):
        start_time = time.time()
        try:
            self.model = model
            self.tokenizer = tokenizer
            device = (
                "cpu"
                if runtime_settings.force_cpu
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            if device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.current_device()}")
            self.gpu_model = model.to(device)
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.processing_batch_size = processing_batch_size
            self.max_workers = max_workers
            
            self.supports_prompts = hasattr(model, 'prompts') and model.prompts is not None
            if self.supports_prompts:
                logger.info(f"Model supports prompts: {list(model.prompts.keys())}")
            else:
                logger.info("Model does not support prompts")
            
            MODEL_LOAD_TIME.observe(time.time() - start_time)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise

    def get_tokenizer(self) -> AutoTokenizer:
        if not hasattr(thread_local, "tokenizer"):
            thread_local.tokenizer = self.tokenizer
        return thread_local.tokenizer

    def split_text_into_chunks(self, text: str) -> list[str]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
            
            if start >= text_length:
                break
        
        return chunks if chunks else [text]

    async def generate_query_embedding(self, text: str) -> list[float]:
        processed_text = preprocess_text(text)

        prompt_name = None
        if (self.supports_prompts and 
            runtime_settings.use_query_prompt and 
            runtime_settings.query_prompt_name):
            if runtime_settings.query_prompt_name in self.model.prompts:
                prompt_name = runtime_settings.query_prompt_name
                logger.debug(f"Using query prompt: {runtime_settings.query_prompt_name}")
            else:
                logger.warning(
                    f"Query prompt '{runtime_settings.query_prompt_name}' not found in model. "
                    f"Available prompts: {list(self.model.prompts.keys())}"
                )

        embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                [processed_text],
                batch_size=1,
                prompt_name=prompt_name,
                normalize_embeddings=True
            ),
        )
        return embedding[0].tolist()

    async def generate_text_embeddings(
        self, texts: dict[int, str]
    ) -> list[TextResponse]:
        if not texts:
            raise ValueError("Empty text dict")

        logger.info(f"Generating embedding for {len(texts)} documents")

        start_time = time.time()

        all_chunks = []
        chunk_counts = []
        chunks_by_id = {}

        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = {
                executor.submit(
                    lambda doc_id_text: (
                        doc_id_text[0],
                        self.split_text_into_chunks(doc_id_text[1]),
                    ),
                    item,
                ): item[0]
                for item in texts.items()
            }

            for future in as_completed(futures):
                doc_id, chunks = future.result()
                CHUNK_COUNT.labels(endpoint="text").inc(len(chunks))
                all_chunks.extend(chunks)
                chunk_counts.append(len(chunks))
                chunks_by_id[doc_id] = chunks

        chunk_time = time.time()
        logger.info(
            f"Producing {len(all_chunks)} chunks took {chunk_time - start_time:.2f} seconds"
        )

        prompt_name = None
        if (self.supports_prompts and 
            runtime_settings.use_document_prompt and 
            runtime_settings.document_prompt_name):
            if runtime_settings.document_prompt_name in self.model.prompts:
                prompt_name = runtime_settings.document_prompt_name
                logger.debug(f"Using document prompt: {runtime_settings.document_prompt_name}")
            else:
                logger.warning(
                    f"Document prompt '{runtime_settings.document_prompt_name}' not found in model. "
                    f"Available prompts: {list(self.model.prompts.keys())}"
                )

        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                all_chunks,
                batch_size=self.processing_batch_size,
                prompt_name=prompt_name
            ),
        )

        embed_time = time.time()
        logger.info(
            f"Generating embedding took {embed_time - chunk_time:.2f} seconds"
        )

        clean_chunks = [
            chunk.replace("", "") for chunk in all_chunks
        ]

        results = []
        embedding_idx = 0

        for doc_id, chunk_count in zip(chunks_by_id.keys(), chunk_counts):
            document_embeddings = embeddings[
                embedding_idx : embedding_idx + chunk_count
            ]
            document_chunks = clean_chunks[
                embedding_idx : embedding_idx + chunk_count
            ]

            doc_results = [
                ChunkEmbedding(
                    chunk_number=idx + 1,
                    chunk=chunk,
                    embedding=embedding.tolist(),
                )
                for idx, (embedding, chunk) in enumerate(
                    zip(document_embeddings, document_chunks)
                )
            ]
            results.append(TextResponse(id=doc_id, embeddings=doc_results))

            embedding_idx += chunk_count

        end_time = time.time()
        logger.info(f"Wrap-up took {end_time - embed_time:.2f} seconds")

        return results

    @staticmethod
    def cleanup_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()