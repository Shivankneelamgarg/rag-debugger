from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable
from uuid import uuid4

from rag_debugger.core.trace import (
    ContextAssembly,
    EmbeddingStep,
    LLMGenerationStep,
    QueryInput,
    RAGTrace,
    RerankedChunk,
    RerankerStep,
    RetrievalStep,
    RetrievedChunk,
    TraceError,
    TracePrivacyConfig,
)


class RAGTracer:
    """Single-trace recorder for local RAG debugging workflows."""

    def __init__(
        self,
        *,
        pipeline_name: str = "default",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        store_full_prompt: bool = False,
        store_full_embeddings: bool = False,
        redacted_fields: list[str] | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.tags = list(tags or [])
        self.metadata = dict(metadata or {})
        self.privacy = TracePrivacyConfig(
            store_full_prompt=store_full_prompt,
            store_full_embeddings=store_full_embeddings,
            redacted_fields=list(redacted_fields or []),
        )
        self._trace: RAGTrace | None = None

    def start_trace(
        self,
        *,
        query_text: str,
        query_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline_name: str | None = None,
        tags: list[str] | None = None,
    ) -> RAGTrace:
        self._trace = RAGTrace(
            pipeline_name=pipeline_name or self.pipeline_name,
            tags=list(tags or self.tags),
            metadata={**self.metadata, **(metadata or {})},
            privacy=self.privacy.model_copy(deep=True),
            query_input=QueryInput(
                text=query_text,
                query_id=query_id or str(uuid4()),
                metadata=metadata or {},
            ),
        )
        return self._trace

    def get_trace(self) -> RAGTrace | None:
        return self._trace

    def get_last_trace(self) -> RAGTrace | None:
        return self._trace

    def _require_trace(self) -> RAGTrace:
        if self._trace is None:
            raise RuntimeError("No active trace. Call start_trace() first.")
        return self._trace

    def record_embedding(
        self,
        *,
        query_text: str,
        embedding_model: str,
        embedding: list[float] | None = None,
        latency_ms: float | None = None,
        tokens_used: int | None = None,
    ) -> EmbeddingStep:
        trace = self._require_trace()
        preview = list((embedding or [])[:8]) if self.privacy.store_full_embeddings else []
        trace.embedding_step = EmbeddingStep(
            query_text=query_text,
            embedding_model=embedding_model,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            embedding_dimensions=len(embedding) if embedding else None,
            embedding_preview=preview,
            embedding=list(embedding) if self.privacy.store_full_embeddings and embedding else None,
        )
        return trace.embedding_step

    def record_retrieval(
        self,
        *,
        retrieved_chunks: list[dict[str, Any] | RetrievedChunk],
        retrieval_method: str,
        top_k: int,
        latency_ms: float | None = None,
        query_embedding: list[float] | None = None,
        reranker_name: str | None = None,
    ) -> RetrievalStep:
        trace = self._require_trace()
        chunks = [
            chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk(**chunk)
            for chunk in retrieved_chunks
        ]
        trace.retrieval_step = RetrievalStep(
            query_embedding=list(query_embedding) if self.privacy.store_full_embeddings and query_embedding else None,
            num_chunks_retrieved=len(chunks),
            retrieved_chunks=chunks,
            retrieval_method=retrieval_method,
            latency_ms=latency_ms,
            top_k=top_k,
            reranker_name=reranker_name,
        )
        return trace.retrieval_step

    def record_context_assembly(
        self,
        *,
        assembled_context: str,
        chunks_used: list[str],
        total_tokens: int | None = None,
        window_size: int | None = None,
        truncation_applied: bool = False,
        latency_ms: float | None = None,
    ) -> ContextAssembly:
        trace = self._require_trace()
        trace.context_assembly = ContextAssembly(
            assembled_context=assembled_context,
            chunks_used=chunks_used,
            total_tokens=total_tokens,
            window_size=window_size,
            truncation_applied=truncation_applied,
            latency_ms=latency_ms,
        )
        return trace.context_assembly

    def record_reranker(
        self,
        *,
        reranker_name: str,
        reranked_chunks: list[dict[str, Any] | RerankedChunk],
        latency_ms: float | None = None,
        input_chunk_count: int | None = None,
    ) -> RerankerStep:
        trace = self._require_trace()
        chunks = [
            chunk if isinstance(chunk, RerankedChunk) else RerankedChunk(**chunk)
            for chunk in reranked_chunks
        ]
        if trace.retrieval_step:
            trace.retrieval_step.reranker_name = reranker_name

        trace.reranker_step = RerankerStep(
            reranker_name=reranker_name,
            latency_ms=latency_ms,
            input_chunk_count=input_chunk_count
            if input_chunk_count is not None
            else (trace.retrieval_step.num_chunks_retrieved if trace.retrieval_step else len(chunks)),
            output_chunk_count=len(chunks),
            reranked_chunks=chunks,
        )
        return trace.reranker_step

    def record_llm_call(
        self,
        *,
        model_name: str,
        generated_answer: str,
        latency_ms: float | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cost_usd: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        user_query: str | None = None,
        full_prompt: str | None = None,
    ) -> LLMGenerationStep:
        trace = self._require_trace()
        trace.llm_generation = LLMGenerationStep(
            model_name=model_name,
            generated_answer=generated_answer,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt if self.privacy.store_full_prompt else None,
            user_query=user_query,
            full_prompt=full_prompt if self.privacy.store_full_prompt else None,
        )
        trace.final_answer = generated_answer
        return trace.llm_generation

    def record_error(self, *, step: str, message: str, error_type: str | None = None) -> TraceError:
        trace = self._require_trace()
        error = TraceError(step=step, message=message, error_type=error_type)
        trace.errors.append(error)
        trace.status = "failed"
        return error

    def finish_trace(self) -> RAGTrace:
        trace = self._require_trace()
        trace.finalize()
        return trace

    @contextmanager
    def trace(
        self,
        *,
        query: str,
        query_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline_name: str | None = None,
        tags: list[str] | None = None,
    ):
        self.start_trace(
            query_text=query,
            query_id=query_id,
            metadata=metadata,
            pipeline_name=pipeline_name,
            tags=tags,
        )
        try:
            yield self
        except Exception as exc:
            self.record_error(step="pipeline", message=str(exc), error_type=type(exc).__name__)
            self.finish_trace()
            raise
        else:
            self.finish_trace()

    def trace_function(
        self,
        func: Callable[..., Any] | None = None,
        *,
        query_arg: str = "query",
        pipeline_name: str | None = None,
        tags: list[str] | None = None,
    ):
        def decorator(wrapped: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(wrapped)
            def inner(*args: Any, **kwargs: Any) -> Any:
                query_text = kwargs.get(query_arg)
                if query_text is None and args:
                    query_text = args[0]
                if not isinstance(query_text, str):
                    raise ValueError(f"Could not resolve string query argument '{query_arg}'.")

                with self.trace(query=query_text, pipeline_name=pipeline_name, tags=tags):
                    result = wrapped(*args, **kwargs)
                    if isinstance(result, str):
                        self.record_llm_call(
                            model_name="unknown",
                            generated_answer=result,
                            user_query=query_text,
                        )
                    return result

            return inner

        if func is None:
            return decorator
        return decorator(func)
