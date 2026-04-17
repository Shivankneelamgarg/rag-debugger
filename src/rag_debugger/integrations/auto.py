from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from rag_debugger.core.tracer import RAGTracer


Embedder = Callable[[str], list[float] | None]
Retriever = Callable[[str, list[float] | None], list[dict[str, Any]]]
Reranker = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
ContextBuilder = Callable[[str, list[dict[str, Any]]], str | dict[str, Any]]
LLMCallable = Callable[[str, str], str | dict[str, Any]]


@dataclass
class AutoInstrumentedRAGPipeline:
    tracer: RAGTracer
    embedder: Embedder
    retriever: Retriever
    llm: LLMCallable
    reranker: Reranker | None = None
    context_builder: ContextBuilder | None = None
    embedding_model: str = "auto-embedder"
    retrieval_method: str = "auto-retriever"
    model_name: str = "auto-llm"
    top_k: int = 5

    def run(self, query: str, *, metadata: dict[str, Any] | None = None) -> Any:
        self.tracer.start_trace(query_text=query, metadata=metadata)

        embedding_start = perf_counter()
        embedding = self.embedder(query)
        embedding_latency_ms = (perf_counter() - embedding_start) * 1000
        self.tracer.record_embedding(
            query_text=query,
            embedding_model=self.embedding_model,
            embedding=embedding,
            latency_ms=embedding_latency_ms,
        )

        retrieval_start = perf_counter()
        retrieved_chunks = self.retriever(query, embedding)
        retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000
        normalized_chunks = self._normalize_chunks(retrieved_chunks)
        self.tracer.record_retrieval(
            retrieved_chunks=normalized_chunks,
            retrieval_method=self.retrieval_method,
            top_k=self.top_k,
            latency_ms=retrieval_latency_ms,
        )

        context_source = normalized_chunks
        if self.reranker is not None:
            rerank_start = perf_counter()
            reranked_chunks = self.reranker(query, normalized_chunks)
            rerank_latency_ms = (perf_counter() - rerank_start) * 1000
            normalized_reranked_chunks = self._normalize_reranked_chunks(reranked_chunks, normalized_chunks)
            self.tracer.record_reranker(
                reranker_name=getattr(self.reranker, "__name__", "auto-reranker"),
                reranked_chunks=normalized_reranked_chunks,
                latency_ms=rerank_latency_ms,
            )
            context_source = reranked_chunks

        context_start = perf_counter()
        context_result = self._build_context(query, context_source)
        context_latency_ms = (perf_counter() - context_start) * 1000
        if isinstance(context_result, dict):
            assembled_context = context_result.get("context", "")
            chunk_ids = context_result.get("chunk_ids") or [item["chunk_id"] for item in context_source]
            total_tokens = context_result.get("total_tokens")
            window_size = context_result.get("window_size")
        else:
            assembled_context = context_result
            chunk_ids = [item["chunk_id"] for item in context_source]
            total_tokens = None
            window_size = None
        self.tracer.record_context_assembly(
            assembled_context=assembled_context,
            chunks_used=chunk_ids,
            total_tokens=total_tokens,
            window_size=window_size,
            latency_ms=context_latency_ms,
        )

        llm_start = perf_counter()
        llm_result = self.llm(query, assembled_context)
        llm_latency_ms = (perf_counter() - llm_start) * 1000
        if isinstance(llm_result, dict):
            self.tracer.record_llm_call(
                model_name=llm_result.get("model_name", self.model_name),
                generated_answer=llm_result.get("answer", ""),
                user_query=query,
                latency_ms=llm_result.get("latency_ms", llm_latency_ms),
                tokens_input=llm_result.get("tokens_input"),
                tokens_output=llm_result.get("tokens_output"),
                cost_usd=llm_result.get("cost_usd"),
            )
        else:
            self.tracer.record_llm_call(
                model_name=self.model_name,
                generated_answer=llm_result,
                user_query=query,
                latency_ms=llm_latency_ms,
            )

        trace = self.tracer.finish_trace()
        return trace

    def _build_context(self, query: str, chunks: list[dict[str, Any]]) -> str | dict[str, Any]:
        if self.context_builder is None:
            return "\n\n".join(chunk.get("text", "") for chunk in chunks)
        return self.context_builder(query, chunks)

    def _normalize_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.get("metadata", {}))
            normalized.append(
                {
                    "chunk_id": chunk.get("chunk_id") or metadata.get("chunk_id") or f"chunk-{index}",
                    "text": chunk.get("text") or chunk.get("page_content") or "",
                    "source": chunk.get("source") or metadata.get("source") or "unknown",
                    "similarity_score": float(chunk.get("similarity_score") or metadata.get("similarity_score") or 0.0),
                    "rank": int(chunk.get("rank") or metadata.get("rank") or index),
                    "chunk_index": int(chunk.get("chunk_index") or metadata.get("chunk_index") or index - 1),
                    "document_id": chunk.get("document_id") or metadata.get("document_id"),
                    "document_version": chunk.get("document_version") or metadata.get("document_version"),
                    "content_hash": chunk.get("content_hash") or metadata.get("content_hash"),
                    "chunk_size": chunk.get("chunk_size") or metadata.get("chunk_size"),
                    "chunk_overlap": chunk.get("chunk_overlap") or metadata.get("chunk_overlap"),
                    "metadata": metadata,
                }
            )
        return normalized

    def _normalize_reranked_chunks(
        self, reranked_chunks: list[dict[str, Any]], retrieved_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rank_lookup = {chunk["chunk_id"]: chunk for chunk in retrieved_chunks}
        normalized: list[dict[str, Any]] = []
        for index, chunk in enumerate(reranked_chunks, start=1):
            chunk_id = chunk.get("chunk_id") or f"chunk-{index}"
            original = rank_lookup.get(chunk_id, {})
            normalized.append(
                {
                    "chunk_id": chunk_id,
                    "original_rank": chunk.get("original_rank") or original.get("rank"),
                    "original_similarity": chunk.get("original_similarity") or original.get("similarity_score"),
                    "reranked_rank": chunk.get("reranked_rank") or index,
                    "reranker_score": chunk.get("reranker_score") or chunk.get("score"),
                }
            )
        return normalized
