# Framework Integration Guides

This document shows how to connect RAG Debugger to the Python app that already calls your model or framework.

RAG Debugger does not plug directly into `chat.openai.com`, `claude.ai`, or `gemini.google.com`. It plugs into the application code around retrieval, context assembly, reranking, and generation.

## Pick The Right Integration Style

- use `RAGTracer` if you want full control and already know where your retrieval and LLM steps live
- use `AutoInstrumentedRAGPipeline` if your RAG app is already structured as Python callables
- use `LangChainTraceAdapter` if your app already works in LangChain document and retriever shapes
- use `LlamaIndexTraceAdapter` if your app already works in LlamaIndex node and query-engine shapes

## OpenAI API

Best for:

- custom Python RAG backends calling OpenAI directly
- apps where you want to log embeddings, request timing, and final answer quality in one place

Minimal pattern:

```python
from openai import OpenAI

from rag_debugger import RAGTracer

client = OpenAI()
tracer = RAGTracer()

query = "When was the company founded?"
tracer.start_trace(query_text=query, metadata={"provider": "openai"})

embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
)
query_embedding = embedding_response.data[0].embedding

tracer.record_embedding(
    query_text=query,
    embedding_model="text-embedding-3-small",
    embedding=query_embedding,
)

retrieved_chunks = [
    {
        "chunk_id": "chunk-1",
        "text": "The company began as a research project in 2018.",
        "source": "docs/about.md",
        "similarity_score": 0.93,
        "rank": 1,
        "chunk_index": 0,
    }
]
tracer.record_retrieval(
    retrieved_chunks=retrieved_chunks,
    retrieval_method="vector_search",
    top_k=5,
)

context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
tracer.record_context_assembly(
    assembled_context=context,
    chunks_used=[chunk["chunk_id"] for chunk in retrieved_chunks],
)

response = client.responses.create(
    model="gpt-4.1",
    instructions="Answer only from the provided context.",
    input=f"Question: {query}\n\nContext:\n{context}",
)
answer = response.output_text

tracer.record_llm_call(
    model_name="gpt-4.1",
    generated_answer=answer,
    user_query=query,
)

trace = tracer.finish_trace()
trace.save("trace.json")
```

Why this is useful:

- you can inspect retrieval quality before blaming the model
- you can compare prompt or chunking changes with `rag-debug diff`
- you can save traces from real production-like traffic and analyze them offline

## Claude API

Best for:

- Anthropic-powered RAG apps using the Messages API
- apps where grounding discipline matters and you want to compare context against the answer

Minimal pattern:

```python
from anthropic import Anthropic

from rag_debugger import RAGTracer

client = Anthropic()
tracer = RAGTracer()

query = "What does the leave policy say?"
tracer.start_trace(query_text=query, metadata={"provider": "anthropic"})

retrieved_chunks = [
    {
        "chunk_id": "policy-1",
        "text": "Employees receive 20 paid leave days each year.",
        "source": "handbook/leave.md",
        "similarity_score": 0.91,
        "rank": 1,
        "chunk_index": 0,
    }
]
tracer.record_retrieval(
    retrieved_chunks=retrieved_chunks,
    retrieval_method="semantic_search",
    top_k=3,
)

context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
tracer.record_context_assembly(
    assembled_context=context,
    chunks_used=["policy-1"],
)

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=300,
    system="Answer only from the retrieved policy context.",
    messages=[
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context}",
        }
    ],
)
answer = "".join(
    block.text for block in message.content if getattr(block, "type", None) == "text"
)

tracer.record_llm_call(
    model_name="claude-sonnet-4-5",
    generated_answer=answer,
    user_query=query,
)

trace = tracer.finish_trace()
trace.save("trace.json")
```

Why this is useful:

- you can catch when Claude answered smoothly but the context was weak
- you can compare retrieval failures versus answer-grounding failures

## Gemini API

Best for:

- Google Gen AI Python apps using `google-genai`
- teams that want system instruction plus retrieved context tracing in one place

Minimal pattern:

```python
from google import genai
from google.genai import types

from rag_debugger import RAGTracer

client = genai.Client()
tracer = RAGTracer()

query = "Summarize the refund policy."
tracer.start_trace(query_text=query, metadata={"provider": "gemini"})

retrieved_chunks = [
    {
        "chunk_id": "refund-1",
        "text": "Refund requests are accepted within 30 days of purchase.",
        "source": "policies/refunds.md",
        "similarity_score": 0.89,
        "rank": 1,
        "chunk_index": 0,
    }
]
tracer.record_retrieval(
    retrieved_chunks=retrieved_chunks,
    retrieval_method="hybrid_search",
    top_k=4,
)

context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
tracer.record_context_assembly(
    assembled_context=context,
    chunks_used=["refund-1"],
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=f"Question: {query}\n\nContext:\n{context}",
    config=types.GenerateContentConfig(
        system_instruction="Answer only from the provided context."
    ),
)

tracer.record_llm_call(
    model_name="gemini-2.0-flash",
    generated_answer=response.text,
    user_query=query,
)

trace = tracer.finish_trace()
trace.save("trace.json")
```

Why this is useful:

- you can separate safety, instruction, and retrieval issues from each other
- you can debug weak evidence before tuning model settings

## LangChain

Best for:

- LangChain apps already using `retriever.invoke(...)` or vector-store retrievers
- teams that want a small adapter instead of rewriting their chain

Minimal pattern:

```python
from rag_debugger import LangChainTraceAdapter, RAGTracer

tracer = RAGTracer()
adapter = LangChainTraceAdapter(tracer)

query = "When was Nike incorporated?"
adapter.start_query(query, metadata={"framework": "langchain"})

documents = retriever.invoke(query)
adapter.on_retrieval(
    documents=[
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ],
    latency_ms=24,
)

context = "\n\n".join(doc.page_content for doc in documents)
adapter.on_context(
    context=context,
    chunk_ids=[
        doc.metadata.get("chunk_id", f"chunk-{index}")
        for index, doc in enumerate(documents, start=1)
    ],
)

answer = llm.invoke(f"Question: {query}\n\nContext:\n{context}")
adapter.on_llm_result(
    model_name="your-langchain-llm",
    answer=getattr(answer, "content", str(answer)),
    user_query=query,
)

trace = adapter.finish()
trace.save("trace.json")
```

Why this is useful:

- you can instrument existing LangChain retrieval without restructuring the whole app
- the adapter converts LangChain documents into the trace format your CLI already understands

## LlamaIndex

Best for:

- LlamaIndex apps using `VectorStoreIndex`, retrievers, or query engines
- teams that want to inspect node scores, reranking, and final answer quality

Minimal pattern:

```python
from rag_debugger import LlamaIndexTraceAdapter, RAGTracer

tracer = RAGTracer()
adapter = LlamaIndexTraceAdapter(tracer)

query = "What did the policy change in 2024?"
adapter.start_query(query, metadata={"framework": "llamaindex"})

nodes = retriever.retrieve(query)
adapter.on_retrieval(
    nodes=[
        {
            "node_id": node.node.node_id,
            "text": node.node.get_content(),
            "score": node.score,
            "metadata": node.node.metadata,
        }
        for node in nodes
    ],
    latency_ms=18,
)

response = query_engine.query(query)
adapter.on_response(
    model_name="your-llamaindex-llm",
    answer=str(response),
    user_query=query,
)

trace = adapter.finish()
trace.save("trace.json")
```

Why this is useful:

- you can keep LlamaIndex indexing and retrieval as-is while tracing the important steps
- you can compare retriever and query-engine changes with saved traces

## Recommended Workflow

For any of the stacks above:

1. instrument one real query path first
2. save a trace with `trace.save("trace.json")`
3. run `rag-debug analyze trace.json`
4. fix one issue at a time
5. compare before and after with `rag-debug diff old.json new.json`

## Official References

These guides were written against current official documentation:

- OpenAI API overview and Responses API: [developers.openai.com/api/reference/overview](https://developers.openai.com/api/reference/overview)
- OpenAI embeddings concepts: [platform.openai.com/docs/introduction](https://platform.openai.com/docs/introduction/)
- Claude getting started and Messages API path: [platform.claude.com/docs/en/get-started](https://platform.claude.com/docs/en/get-started)
- Gemini `generate_content` and system instruction: [ai.google.dev/api/generate-content](https://ai.google.dev/api/generate-content)
- LangChain retriever usage: [docs.langchain.com/oss/python/langchain/knowledge-base](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- LlamaIndex `VectorStoreIndex` guide: [developers.llamaindex.ai/python/framework/module_guides/indexing/vector_store_index](https://developers.llamaindex.ai/python/framework/module_guides/indexing/vector_store_index/)
