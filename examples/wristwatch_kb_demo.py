from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag_debugger import RAGTracer


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set[str]:
    text = normalize_text(text).lower()
    # simple word tokenizer (keeps numbers)
    return set(re.findall(r"[a-z0-9]+", text))


def overlap_score(query: str, doc: str) -> float:
    q = tokenize(query)
    d = tokenize(doc)
    if not q or not d:
        return 0.0
    inter = len(q & d)
    return inter / max(1, len(q))


def first_sentence(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    # Split on sentence-ish boundaries; keep it conservative.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a RAG Debugger trace from a simple JSON knowledge base.")
    parser.add_argument(
        "--kb",
        required=True,
        help="Path to a knowledge base JSON file (must contain a `chunks` list with `content`).",
    )
    parser.add_argument("--query", default="watch 5 min forward good time", help="Query to test retrieval.")
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve into the trace.")
    parser.add_argument(
        "--out",
        default="examples/generated/wristwatch_kb_trace.json",
        help="Where to write the generated trace JSON.",
    )
    args = parser.parse_args()

    kb_path = Path(args.kb)
    data = json.loads(kb_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit("Knowledge base JSON must have a non-empty `chunks` list.")

    query = args.query

    scored: list[tuple[float, int, dict]] = []
    for idx, item in enumerate(chunks):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        score = overlap_score(query, content)
        scored.append((score, idx, item))

    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    top = scored[: max(1, int(args.top_k))]

    tracer = RAGTracer()
    tracer.start_trace(
        query_text=query,
        pipeline_name="wristwatch-kb-demo",
        metadata={
            "kb_source": data.get("source"),
            "kb_topic": data.get("topic"),
            "kb_total_chunks": data.get("total_chunks") or len(chunks),
        },
        tags=["example", "kb"],
    )

    retrieved_chunks = []
    for rank, (score, idx, item) in enumerate(top, start=1):
        content = normalize_text(str(item.get("content") or ""))
        retrieved_chunks.append(
            {
                "chunk_id": f"kb-{idx}",
                "text": content,
                "source": str(data.get("source") or "knowledge_base"),
                "similarity_score": round(float(score), 3),
                "rank": rank,
                "chunk_index": idx,
                "metadata": {
                    "section": item.get("section"),
                    "day": item.get("day"),
                },
            }
        )

    tracer.record_retrieval(
        retrieved_chunks=retrieved_chunks,
        retrieval_method="keyword_overlap_demo",
        top_k=len(retrieved_chunks),
        latency_ms=5,
    )

    assembled_context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in retrieved_chunks)
    tracer.record_context_assembly(
        assembled_context=assembled_context,
        chunks_used=[c["chunk_id"] for c in retrieved_chunks],
        total_tokens=None,
        window_size=4096,
        truncation_applied=False,
        latency_ms=2,
    )

    # Offline answer: take a sentence from the top chunk so grounding stays clean.
    top_sentence = first_sentence(retrieved_chunks[0]["text"]) if retrieved_chunks else ""
    answer = top_sentence or "No relevant content found in the knowledge base."

    tracer.record_llm_call(
        model_name="offline-demo",
        generated_answer=answer,
        user_query=query,
        latency_ms=10,
        tokens_input=None,
        tokens_output=None,
        cost_usd=None,
    )

    trace = tracer.finish_trace()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.save(out_path)

    print("Wrote trace:", out_path)
    print("Next:")
    print(f"  rag-debug analyze {out_path}")
    print(f"  rag-debug export {out_path} --format html --output report.html")


if __name__ == "__main__":
    main()
