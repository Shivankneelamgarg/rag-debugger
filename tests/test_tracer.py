import pytest

from rag_debugger import RAGTracer


def test_manual_trace_flow() -> None:
    tracer = RAGTracer(store_full_embeddings=False, store_full_prompt=False)
    tracer.start_trace(query_text="What is RAG?")
    tracer.record_embedding(
        query_text="What is RAG?",
        embedding_model="test-embed",
        embedding=[0.1, 0.2, 0.3],
        latency_ms=12,
    )
    tracer.record_retrieval(
        retrieved_chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "RAG retrieves documents before generation.",
                "source": "docs/1.md",
                "similarity_score": 0.93,
                "rank": 1,
                "chunk_index": 0,
            }
        ],
        retrieval_method="semantic_search",
        top_k=3,
        latency_ms=18,
    )
    tracer.record_context_assembly(
        assembled_context="RAG retrieves documents before generation.",
        chunks_used=["chunk-1"],
        total_tokens=25,
        window_size=1024,
        latency_ms=5,
    )
    tracer.record_llm_call(
        model_name="gpt-4.1-mini",
        generated_answer="RAG retrieves documents before generation.",
        full_prompt="system + user",
        user_query="What is RAG?",
        latency_ms=44,
    )
    trace = tracer.finish_trace()

    assert trace.status == "completed"
    assert trace.embedding_step.embedding is None
    assert trace.embedding_step.embedding_preview == []
    assert trace.llm_generation.full_prompt is None
    assert trace.summary.total_latency_ms == 79


def test_context_manager_records_error() -> None:
    tracer = RAGTracer()
    with pytest.raises(RuntimeError):
        with tracer.trace(query="explode"):
            raise RuntimeError("boom")

    trace = tracer.get_trace()
    assert trace is not None
    assert trace.status == "failed"
    assert trace.errors[0].message == "boom"


def test_decorator_flow() -> None:
    tracer = RAGTracer()

    @tracer.trace_function
    def pipeline(query: str) -> str:
        return f"Answer: {query}"

    result = pipeline("hello")
    trace = tracer.get_last_trace()
    assert result == "Answer: hello"
    assert trace is not None
    assert trace.final_answer == "Answer: hello"


def test_finish_trace_records_validation_warning() -> None:
    tracer = RAGTracer()
    tracer.start_trace(query_text="What is RAG?")
    tracer.record_retrieval(
        retrieved_chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "RAG retrieves documents before generation.",
                "source": "docs/1.md",
                "similarity_score": 0.93,
                "rank": 1,
                "chunk_index": 0,
            }
        ],
        retrieval_method="semantic_search",
        top_k=3,
        latency_ms=18,
    )
    tracer.record_context_assembly(
        assembled_context="RAG retrieves documents before generation.",
        chunks_used=["chunk-missing"],
        latency_ms=7,
    )

    trace = tracer.finish_trace()
    assert trace.summary.total_latency_ms == 25
    assert trace.summary.validation_warnings


def test_record_reranker_step() -> None:
    tracer = RAGTracer()
    tracer.start_trace(query_text="What is RAG?")
    tracer.record_retrieval(
        retrieved_chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "RAG retrieves documents before generation.",
                "source": "docs/1.md",
                "similarity_score": 0.93,
                "rank": 1,
                "chunk_index": 0,
            },
            {
                "chunk_id": "chunk-2",
                "text": "RAG reduces unsupported claims with context.",
                "source": "docs/2.md",
                "similarity_score": 0.88,
                "rank": 2,
                "chunk_index": 1,
            },
        ],
        retrieval_method="semantic_search",
        top_k=3,
        latency_ms=18,
    )
    tracer.record_reranker(
        reranker_name="cross-encoder-mini",
        latency_ms=6,
        reranked_chunks=[
            {"chunk_id": "chunk-2", "original_rank": 2, "reranked_rank": 1, "reranker_score": 0.97},
            {"chunk_id": "chunk-1", "original_rank": 1, "reranked_rank": 2, "reranker_score": 0.94},
        ],
    )

    trace = tracer.finish_trace()
    assert trace.reranker_step is not None
    assert trace.reranker_step.reranker_name == "cross-encoder-mini"
    assert trace.retrieval_step.reranker_name == "cross-encoder-mini"
    assert trace.summary.total_latency_ms == 24
