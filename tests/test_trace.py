from pathlib import Path

from rag_debugger.core.trace import (
    ContextAssembly,
    LLMGenerationStep,
    QueryInput,
    RAGTrace,
    RerankedChunk,
    RerankerStep,
    RetrievalStep,
    RetrievedChunk,
)


def make_trace() -> RAGTrace:
    trace = RAGTrace(
        pipeline_name="demo",
        query_input=QueryInput(text="What is RAG?"),
        retrieval_step=RetrievalStep(
            num_chunks_retrieved=2,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    text="RAG grounds answers with retrieved documents.",
                    source="docs/a.md",
                    similarity_score=0.94,
                    rank=1,
                    chunk_index=0,
                ),
                RetrievedChunk(
                    chunk_id="chunk-2",
                    text="RAG also improves factual consistency.",
                    source="docs/b.md",
                    similarity_score=0.89,
                    rank=2,
                    chunk_index=0,
                ),
            ],
            retrieval_method="semantic_search",
            top_k=5,
            latency_ms=30,
        ),
        context_assembly=ContextAssembly(
            assembled_context="RAG grounds answers with retrieved documents.",
            chunks_used=["chunk-1"],
            total_tokens=40,
            window_size=512,
        ),
        llm_generation=LLMGenerationStep(
            model_name="gpt-4.1-mini",
            generated_answer="RAG uses retrieved documents to ground an answer.",
            latency_ms=250,
            tokens_input=130,
            tokens_output=40,
        ),
        final_answer="RAG uses retrieved documents to ground an answer.",
    )
    return trace.finalize()


def test_trace_round_trip(tmp_path: Path) -> None:
    trace = make_trace()
    target = tmp_path / "trace.json"
    trace.save(target)

    loaded = RAGTrace.load(target)
    assert loaded.trace_id == trace.trace_id
    assert loaded.summary.total_latency_ms == 280
    assert loaded.final_answer == trace.final_answer


def test_batch_round_trip(tmp_path: Path) -> None:
    traces = [make_trace(), make_trace()]
    target = tmp_path / "traces.jsonl"
    RAGTrace.save_batch(traces, target)

    loaded = RAGTrace.load_batch(target)
    assert len(loaded) == 2
    assert loaded[0].query_input.text == "What is RAG?"


def test_trace_round_trip_with_reranker(tmp_path: Path) -> None:
    trace = make_trace()
    trace.reranker_step = RerankerStep(
        reranker_name="cross-encoder-mini",
        latency_ms=11,
        input_chunk_count=2,
        output_chunk_count=2,
        reranked_chunks=[
            RerankedChunk(chunk_id="chunk-2", original_rank=2, reranked_rank=1, reranker_score=0.96),
            RerankedChunk(chunk_id="chunk-1", original_rank=1, reranked_rank=2, reranker_score=0.94),
        ],
    )
    trace.finalize()
    target = tmp_path / "reranked.json"
    trace.save(target)

    loaded = RAGTrace.load(target)
    assert loaded.reranker_step is not None
    assert loaded.reranker_step.reranked_chunks[0].chunk_id == "chunk-2"
