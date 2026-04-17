from __future__ import annotations

from typing import Any

from rag_debugger.core.tracer import RAGTracer


class LangChainTraceAdapter:
    """Small adapter for wiring LangChain-style events into RAGTracer."""

    def __init__(self, tracer: RAGTracer) -> None:
        self.tracer = tracer

    def start_query(self, query: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.tracer.start_trace(query_text=query, metadata=metadata)

    def on_embedding(
        self,
        *,
        query_text: str,
        embedding_model: str,
        embedding: list[float] | None = None,
        latency_ms: float | None = None,
        tokens_used: int | None = None,
    ) -> None:
        self.tracer.record_embedding(
            query_text=query_text,
            embedding_model=embedding_model,
            embedding=embedding,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )

    def on_retrieval(
        self,
        *,
        documents: list[dict[str, Any]],
        retrieval_method: str = "langchain_retriever",
        top_k: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        chunks: list[dict[str, Any]] = []
        for index, doc in enumerate(documents, start=1):
            metadata = dict(doc.get("metadata", {}))
            chunks.append(
                {
                    "chunk_id": doc.get("chunk_id") or metadata.get("chunk_id") or f"chunk-{index}",
                    "text": doc.get("page_content") or doc.get("text") or "",
                    "source": metadata.get("source") or doc.get("source") or "unknown",
                    "similarity_score": float(doc.get("similarity_score") or metadata.get("similarity_score") or 0.0),
                    "rank": int(doc.get("rank") or metadata.get("rank") or index),
                    "chunk_index": int(metadata.get("chunk_index") or index - 1),
                    "document_id": metadata.get("document_id"),
                    "document_version": metadata.get("document_version"),
                    "content_hash": metadata.get("content_hash"),
                    "chunk_size": metadata.get("chunk_size"),
                    "chunk_overlap": metadata.get("chunk_overlap"),
                    "metadata": metadata,
                }
            )

        self.tracer.record_retrieval(
            retrieved_chunks=chunks,
            retrieval_method=retrieval_method,
            top_k=top_k or len(chunks),
            latency_ms=latency_ms,
        )

    def on_context(self, *, context: str, chunk_ids: list[str], total_tokens: int | None = None) -> None:
        self.tracer.record_context_assembly(
            assembled_context=context,
            chunks_used=chunk_ids,
            total_tokens=total_tokens,
        )

    def on_llm_result(
        self,
        *,
        model_name: str,
        answer: str,
        user_query: str | None = None,
        latency_ms: float | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        self.tracer.record_llm_call(
            model_name=model_name,
            generated_answer=answer,
            user_query=user_query,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
        )

    def finish(self):
        return self.tracer.finish_trace()
