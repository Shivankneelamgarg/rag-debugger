from rag_debugger import (
    AutoInstrumentedRAGPipeline,
    ExternalExplanationProvider,
    LangChainTraceAdapter,
    LlamaIndexTraceAdapter,
    RAGTracer,
    explain_trace,
)


def test_langchain_adapter_flow() -> None:
    tracer = RAGTracer()
    adapter = LangChainTraceAdapter(tracer)
    adapter.start_query("What is RAG?")
    adapter.on_retrieval(
        documents=[
            {
                "page_content": "RAG grounds answers with retrieved context.",
                "similarity_score": 0.92,
                "metadata": {"source": "docs/rag.md", "chunk_id": "chunk-1"},
            }
        ]
    )
    adapter.on_context(context="RAG grounds answers with retrieved context.", chunk_ids=["chunk-1"])
    adapter.on_llm_result(model_name="claude-sonnet", answer="RAG grounds answers.", user_query="What is RAG?")
    trace = adapter.finish()
    assert trace.retrieval_step is not None
    assert trace.final_answer == "RAG grounds answers."


def test_llamaindex_adapter_flow() -> None:
    tracer = RAGTracer()
    adapter = LlamaIndexTraceAdapter(tracer)
    adapter.start_query("What is RAG?")
    adapter.on_retrieval(
        nodes=[
            {
                "node_id": "node-1",
                "text": "RAG retrieves useful context.",
                "score": 0.88,
                "metadata": {"source": "docs/rag.md"},
            }
        ]
    )
    adapter.on_rerank(
        reranker_name="bge-reranker",
        nodes=[{"node_id": "node-1", "original_rank": 1, "score": 0.97}],
    )
    adapter.on_response(model_name="gemini-flash", answer="RAG retrieves useful context.")
    trace = adapter.finish()
    assert trace.reranker_step is not None
    assert trace.reranker_step.reranker_name == "bge-reranker"


def test_explain_trace_returns_summary() -> None:
    tracer = RAGTracer()
    tracer.start_trace(query_text="What is RAG?")
    tracer.record_llm_call(model_name="gpt-4.1-mini", generated_answer="RAG retrieves documents.")
    trace = tracer.finish_trace()
    explanation = explain_trace(trace)
    assert "Overall health" in explanation


def test_auto_instrumented_pipeline() -> None:
    tracer = RAGTracer()
    pipeline = AutoInstrumentedRAGPipeline(
        tracer=tracer,
        embedder=lambda query: [0.1, 0.2, 0.3],
        retriever=lambda query, embedding: [
            {
                "chunk_id": "chunk-1",
                "text": "RAG retrieves documents.",
                "source": "docs/rag.md",
                "similarity_score": 0.93,
                "rank": 1,
                "chunk_index": 0,
            }
        ],
        reranker=lambda query, docs: [
            {"chunk_id": "chunk-1", "original_rank": 1, "reranked_rank": 1, "reranker_score": 0.98}
        ],
        llm=lambda query, context: {"answer": "RAG retrieves documents.", "model_name": "gpt-4.1-mini"},
    )
    trace = pipeline.run("What is RAG?")
    assert trace.embedding_step is not None
    assert trace.reranker_step is not None
    assert trace.final_answer == "RAG retrieves documents."


def test_external_explanation_provider() -> None:
    tracer = RAGTracer()
    tracer.start_trace(query_text="What is RAG?")
    tracer.record_llm_call(model_name="gpt-4.1-mini", generated_answer="RAG retrieves documents.")
    trace = tracer.finish_trace()
    provider = ExternalExplanationProvider(lambda trace, analysis: f"custom:{analysis.status}")
    explanation = explain_trace(trace, provider=provider)
    assert explanation.startswith("custom:")
