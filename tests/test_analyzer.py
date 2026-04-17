from rag_debugger.analysis.analyzer import HealthWeights
from rag_debugger.core.trace import (
    ContextAssembly,
    LLMGenerationStep,
    QueryInput,
    RAGTrace,
    RetrievalStep,
    RetrievedChunk,
)


def test_analyzer_flags_bad_chunk_and_truncation() -> None:
    trace = RAGTrace(
        pipeline_name="bad-pipeline",
        query_input=QueryInput(text="When was the company founded?"),
        retrieval_step=RetrievalStep(
            num_chunks_retrieved=2,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="good",
                    text="The company started as a research lab in 2018.",
                    source="a.md",
                    similarity_score=0.91,
                    rank=1,
                    chunk_index=0,
                ),
                RetrievedChunk(
                    chunk_id="bad",
                    text="The office kitchen serves coffee every morning.",
                    source="b.md",
                    similarity_score=0.42,
                    rank=2,
                    chunk_index=0,
                ),
            ],
            retrieval_method="semantic_search",
            top_k=4,
            latency_ms=55,
        ),
        context_assembly=ContextAssembly(
            assembled_context="The office kitchen serves coffee every morning.",
            chunks_used=["bad"],
            total_tokens=980,
            window_size=1000,
            truncation_applied=True,
        ),
        llm_generation=LLMGenerationStep(
            model_name="gpt-4.1-mini",
            generated_answer="The company was founded in 2020. It serves coffee every morning.",
            latency_ms=3200,
            tokens_input=4800,
            tokens_output=320,
        ),
        final_answer="The company was founded in 2020. It serves coffee every morning.",
    ).finalize()

    analysis = trace.analyze(weights=HealthWeights())
    assert analysis.retrieval.score < 0.8
    assert analysis.context.truncation_applied is True
    assert analysis.grounding.score < 1.0
    assert any("Filter low-similarity chunks" in item for item in analysis.recommendations)


def test_partial_trace_analyzes_cleanly() -> None:
    trace = RAGTrace(
        query_input=QueryInput(text="What is RAG?"),
        final_answer="RAG helps models use retrieved data.",
    ).finalize()

    analysis = trace.analyze()
    assert analysis.overall_health > 0


def test_grounding_rejects_conflicting_numeric_claims() -> None:
    trace = RAGTrace(
        query_input=QueryInput(text="When was the company founded?"),
        context_assembly=ContextAssembly(
            assembled_context="The company was founded in 2018.",
            chunks_used=["chunk-1"],
        ),
        final_answer="The company was founded in 2020.",
    ).finalize()

    analysis = trace.analyze()
    assert analysis.grounding.score == 0.0
    assert analysis.grounding.unsupported_claims == ["The company was founded in 2020"]
