from __future__ import annotations

from typing import Any

from rag_debugger.core.tracer import RAGTracer


class LlamaIndexTraceAdapter:
    """Small adapter for wiring LlamaIndex-style events into RAGTracer."""

    def __init__(self, tracer: RAGTracer) -> None:
        self.tracer = tracer

    def start_query(self, query: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.tracer.start_trace(query_text=query, metadata=metadata)

    def on_retrieval(
        self,
        *,
        nodes: list[dict[str, Any]],
        retrieval_method: str = "llamaindex_retriever",
        latency_ms: float | None = None,
    ) -> None:
        chunks: list[dict[str, Any]] = []
        for index, node in enumerate(nodes, start=1):
            metadata = dict(node.get("metadata", {}))
            chunks.append(
                {
                    "chunk_id": node.get("node_id") or metadata.get("chunk_id") or f"chunk-{index}",
                    "text": node.get("text") or "",
                    "source": metadata.get("source") or "unknown",
                    "similarity_score": float(node.get("score") or metadata.get("similarity_score") or 0.0),
                    "rank": index,
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
            top_k=len(chunks),
            latency_ms=latency_ms,
        )

    def on_rerank(self, *, reranker_name: str, nodes: list[dict[str, Any]], latency_ms: float | None = None) -> None:
        reranked_chunks = []
        for index, node in enumerate(nodes, start=1):
            reranked_chunks.append(
                {
                    "chunk_id": node.get("node_id") or f"chunk-{index}",
                    "original_rank": node.get("original_rank"),
                    "reranked_rank": index,
                    "reranker_score": node.get("score"),
                }
            )
        self.tracer.record_reranker(
            reranker_name=reranker_name,
            reranked_chunks=reranked_chunks,
            latency_ms=latency_ms,
        )

    def on_response(
        self,
        *,
        model_name: str,
        answer: str,
        user_query: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        self.tracer.record_llm_call(
            model_name=model_name,
            generated_answer=answer,
            user_query=user_query,
            latency_ms=latency_ms,
        )

    def finish(self):
        return self.tracer.finish_trace()
