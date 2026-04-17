# RAG Debugger Status

This file tracks what is already implemented so progress is always clear.

## Completed

### Phase 0: Project Setup

- `pyproject.toml` added with packaging, dependencies, and `rag-debug` CLI entrypoint
- `src/rag_debugger/` package created
- `tests/` added
- `examples/` added
- `README.md`, `LICENSE`, and GitHub issue templates added

### Phase 1: Trace Schema and Storage

- `QueryInput`, `EmbeddingStep`, `RetrievedChunk`, `RetrievalStep`, `ContextAssembly`, `LLMGenerationStep`, and `RAGTrace` implemented
- `status`, `errors`, `metadata`, privacy config, and summary fields implemented
- JSON save and load implemented
- JSONL batch save and load implemented
- Failed traces can still be saved and inspected

### Phase 2: Tracer API

- `RAGTracer.start_trace(...)`
- `record_embedding(...)`
- `record_retrieval(...)`
- `record_context_assembly(...)`
- `record_llm_call(...)`
- `record_error(...)`
- `finish_trace()`
- context manager tracing
- decorator tracing

### Phase 3: Analysis Engine

- retrieval quality scoring
- context completeness checks
- heuristic grounding analysis
- performance scoring
- configurable health weights
- deterministic recommendations

### Phase 4: CLI

- `rag-debug view trace.json`
- `rag-debug analyze trace.json`
- `rag-debug stats traces/`
- `rag-debug export trace.json --format json|csv|pretty`

### Phase 5: Testing and Release Basics

- model and serialization tests
- tracer flow tests
- analyzer tests
- CLI tests
- sample traces for healthy, weak retrieval, and hallucination-style cases

## Quality Hardening Added

- embedding previews are no longer stored unless full embedding storage is enabled
- context assembly latency is now included in total latency
- trace finalization now warns when context references chunks not present in retrieval results
- grounding checks now require numeric claims to match numeric evidence in context
- CLI `view` now shows validation warnings when trace consistency issues exist

## Completed

### Phase 6: Post-v1 Improvements

- reranker step support
- `rag-debug diff` for side-by-side trace comparison
- HTML report export through `rag-debug export --format html`
- LangChain adapter helpers
- LlamaIndex adapter helpers
- aggregate analytics command
- static dashboard generation command
- explanation command with optional custom generator
- auto-instrumented pipeline wrapper for deeper framework instrumentation
- live auto-refreshing dashboard server
- richer team report analytics with tags, grouped metrics, daily trend, and top findings
- structured and external explanation providers

## Verified

- `pip install -e ".[dev]"` works
- `rag-debug --help` works
- `pytest` passes

## Current Bottom Line

This project is not only Phase 1 complete.

It currently has Phases 0 through 6 implemented in a usable v1 form.
