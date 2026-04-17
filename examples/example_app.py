from __future__ import annotations

from pathlib import Path

from rag_debugger import AutoInstrumentedRAGPipeline, RAGTracer


DATASET = [
    {
        "chunk_id": "company-history",
        "document_id": "about-page",
        "text": "The company was founded as a research project in 2018 and launched commercially in 2020.",
        "source": "docs/about.md",
        "metadata": {"team": "company", "topic": "history"},
    },
    {
        "chunk_id": "leave-policy",
        "document_id": "handbook",
        "text": "The handbook leave policy says employees receive 12 days of annual leave each year. Unused leave does not roll over automatically.",
        "source": "docs/handbook.md",
        "metadata": {"team": "people", "topic": "policy"},
    },
    {
        "chunk_id": "support-hours",
        "document_id": "support-page",
        "text": "Customer support is available from 9 AM to 6 PM on weekdays.",
        "source": "docs/support.md",
        "metadata": {"team": "support", "topic": "hours"},
    },
]


def fake_embedder(query: str) -> list[float]:
    # Small deterministic embedding purely for demo purposes.
    base = float((sum(ord(char) for char in query) % 100) / 100)
    return [round(base, 2), round(base / 2 + 0.1, 2), round(base / 3 + 0.2, 2)]


def fake_retriever(query: str, embedding: list[float] | None) -> list[dict]:
    query_lower = query.lower()
    stopwords = {"the", "a", "an", "is", "was", "what", "when", "does", "about"}
    keywords = [word.strip("?,.!") for word in query_lower.split() if word.strip("?,.!") not in stopwords]
    scored: list[dict] = []
    for index, item in enumerate(DATASET, start=1):
        text_lower = item["text"].lower()
        score = 0.15
        for keyword in keywords:
            if keyword in text_lower:
                score += 0.35
        score = min(score, 0.98)
        scored.append(
            {
                **item,
                "similarity_score": round(score, 2),
                "rank": index,
                "chunk_index": index - 1,
            }
        )
    scored.sort(key=lambda chunk: chunk["similarity_score"], reverse=True)
    for index, chunk in enumerate(scored, start=1):
        chunk["rank"] = index
    return scored[:3]


def fake_reranker(query: str, docs: list[dict]) -> list[dict]:
    query_lower = query.lower()
    reranked = sorted(
        docs,
        key=lambda chunk: (
            "leave" in query_lower and "leave" in chunk["text"].lower(),
            "founded" in query_lower and "2018" in chunk["text"],
            chunk.get("similarity_score", 0.0),
        ),
        reverse=True,
    )
    normalized = []
    for index, chunk in enumerate(reranked, start=1):
        normalized.append(
            {
                **chunk,
                "chunk_id": chunk["chunk_id"],
                "original_rank": chunk.get("rank"),
                "reranked_rank": index,
                "reranker_score": round(chunk.get("similarity_score", 0.0) + 0.02, 2),
            }
        )
    return normalized


def fake_context_builder(query: str, docs: list[dict]) -> dict:
    chosen = docs[:1]
    context = "\n\n".join(doc["text"] for doc in chosen)
    return {
        "context": context,
        "chunk_ids": [doc["chunk_id"] for doc in chosen],
        "total_tokens": len(context.split()),
        "window_size": 4096,
    }


def fake_llm(query: str, context: str) -> dict:
    query_lower = query.lower()
    if "founded" in query_lower:
        answer = "The company began as a research project in 2018 and launched commercially in 2020."
    elif "leave" in query_lower:
        answer = "Employees receive 12 days of annual leave and unused leave does not roll over automatically."
    else:
        answer = "Customer support is available from 9 AM to 6 PM on weekdays."

    return {
        "model_name": "demo-llm",
        "answer": answer,
        "tokens_input": len(context.split()) + len(query.split()),
        "tokens_output": len(answer.split()),
        "cost_usd": 0.0004,
    }


def run_demo(query: str, output_dir: Path) -> Path:
    tracer = RAGTracer(pipeline_name="example-demo", tags=["example", "demo-app"])
    pipeline = AutoInstrumentedRAGPipeline(
        tracer=tracer,
        embedder=fake_embedder,
        retriever=fake_retriever,
        reranker=fake_reranker,
        context_builder=fake_context_builder,
        llm=fake_llm,
        embedding_model="demo-embedder",
        retrieval_method="demo-semantic-search",
        model_name="demo-llm",
        top_k=3,
    )
    trace = pipeline.run(query)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = query.lower().replace(" ", "_").replace("?", "").replace("/", "_")
    path = output_dir / f"{filename}.json"
    trace.save(path)
    return path


def main() -> None:
    output_dir = Path(__file__).parent / "generated"
    display_dir = "examples/generated"
    queries = [
        "When was the company founded?",
        "What does the handbook say about leave policy?",
    ]

    print("Running demo RAG app with built-in fake data...")
    generated_paths = [run_demo(query, output_dir) for query in queries]

    print("\nGenerated traces:")
    for path in generated_paths:
        print(f"- {display_dir}/{path.name}")

    print("\nNext commands to try:")
    print(f"  rag-debug analyze {display_dir}/{generated_paths[0].name}")
    print(f"  rag-debug analyze {display_dir}/{generated_paths[1].name}")
    print(f"  rag-debug diff {display_dir}/{generated_paths[0].name} {display_dir}/{generated_paths[1].name}")
    print(f"  rag-debug export {display_dir}/{generated_paths[0].name} --format html --output {display_dir}/report.html")
    print("  rag-debug team-report examples/generated --group-by tag")


if __name__ == "__main__":
    main()
