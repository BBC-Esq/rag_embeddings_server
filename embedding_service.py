# embedding_service.py
import asyncio
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import runtime_settings
from metrics import CHUNK_COUNT, MODEL_LOAD_TIME
from schemas import ChunkEmbedding, TextResponse
from utils import (
    download_nltk_resources,
    logger,
    preprocess_text,
    verify_nltk_resources,
)

torch.set_float32_matmul_precision("high")
thread_local = threading.local()


class EmbeddingService:
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: AutoTokenizer,
        max_tokens: int,
        overlap_ratio: float,
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
            self.max_tokens = max_tokens
            self.num_overlap_sentences = int(max_tokens * overlap_ratio)
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

    @staticmethod
    def handle_sent_tokenize(text: str) -> list[str]:
        try:
            return sent_tokenize(text)
        except (zipfile.BadZipFile, LookupError):
            download_nltk_resources()
            try:
                verify_nltk_resources()
                return sent_tokenize(text)
            except Exception as e:
                logger.error(
                    f"Sentence tokenization failed after retry: {str(e)}"
                )
                raise

    def split_text_into_chunks(self, text: str) -> list[str]:
        sentences = self.handle_sent_tokenize(text)
        local_tokenizer = self.get_tokenizer()
        encoded_sentences = [
            local_tokenizer.encode(sentence, add_special_tokens=False)
            for sentence in sentences
        ]
        lead_text = ""
        lead_tokens = [] if not lead_text else local_tokenizer.encode(lead_text)
        lead_len = len(lead_tokens)
        chunks = []
        current_chunks: list[str] = []
        current_token_counts = len(lead_tokens)

        for sentence_tokens in encoded_sentences:
            sentence_len = len(sentence_tokens)
            if lead_len + sentence_len > self.max_tokens:
                if current_chunks:
                    chunks.append(lead_text + " ".join(current_chunks))
                truncated_sentence = local_tokenizer.decode(
                    sentence_tokens[: (self.max_tokens - len(lead_tokens))]
                )
                chunks.append(lead_text + truncated_sentence)

                current_chunks = []
                current_token_counts = lead_len
                continue

            if current_token_counts + sentence_len > self.max_tokens:
                overlap_sentences = current_chunks[
                    -max(0, self.num_overlap_sentences) :
                ]
                if current_chunks:
                    chunks.append(lead_text + " ".join(current_chunks))

                overlap_token_counts = local_tokenizer.encode(
                    " ".join(overlap_sentences), add_special_tokens=False
                )
                if (
                    lead_len + len(overlap_token_counts) + sentence_len
                    > self.max_tokens
                ):
                    current_chunks = [local_tokenizer.decode(sentence_tokens)]
                    current_token_counts = lead_len + sentence_len
                else:
                    current_chunks = overlap_sentences + [
                        local_tokenizer.decode(sentence_tokens)
                    ]
                    current_token_counts = (
                        lead_len + len(overlap_token_counts) + sentence_len
                    )
                continue

            current_chunks.append(local_tokenizer.decode(sentence_tokens))
            current_token_counts += len(sentence_tokens)

        if current_chunks:
            chunks.append(lead_text + " ".join(current_chunks))
        return chunks

    async def generate_query_embedding(self, text: str) -> list[float]:
        processed_text = preprocess_text(text)

        encode_kwargs = {
            "sentences": [processed_text],
            "batch_size": 1
        }
        
        if (self.supports_prompts and 
            runtime_settings.use_query_prompt and 
            runtime_settings.query_prompt_name):
            if runtime_settings.query_prompt_name in self.model.prompts:
                encode_kwargs["prompt_name"] = runtime_settings.query_prompt_name
                logger.debug(f"Using query prompt: {runtime_settings.query_prompt_name}")
            else:
                logger.warning(
                    f"Query prompt '{runtime_settings.query_prompt_name}' not found in model. "
                    f"Available prompts: {list(self.model.prompts.keys())}"
                )

        embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(**encode_kwargs),
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

        encode_kwargs = {
            "sentences": all_chunks,
            "batch_size": self.processing_batch_size
        }
        
        if (self.supports_prompts and 
            runtime_settings.use_document_prompt and 
            runtime_settings.document_prompt_name):
            if runtime_settings.document_prompt_name in self.model.prompts:
                encode_kwargs["prompt_name"] = runtime_settings.document_prompt_name
                logger.debug(f"Using document prompt: {runtime_settings.document_prompt_name}")
            else:
                logger.warning(
                    f"Document prompt '{runtime_settings.document_prompt_name}' not found in model. "
                    f"Available prompts: {list(self.model.prompts.keys())}"
                )

        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(**encode_kwargs),
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