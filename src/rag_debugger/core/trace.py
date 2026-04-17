from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


TraceStatus = Literal["pending", "completed", "failed"]


class QueryInput(BaseModel):
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingStep(BaseModel):
    query_text: str
    embedding_model: str
    latency_ms: float | None = None
    tokens_used: int | None = None
    embedding_dimensions: int | None = None
    embedding_preview: list[float] = Field(default_factory=list)
    embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    similarity_score: float
    rank: int
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str | None = None
    document_version: str | None = None
    content_hash: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class RetrievalStep(BaseModel):
    query_embedding: list[float] | None = None
    num_chunks_retrieved: int
    retrieved_chunks: list[RetrievedChunk]
    retrieval_method: str
    latency_ms: float | None = None
    top_k: int
    reranker_name: str | None = None


class RerankedChunk(BaseModel):
    chunk_id: str
    reranked_rank: int
    original_rank: int | None = None
    original_similarity: float | None = None
    reranker_score: float | None = None


class RerankerStep(BaseModel):
    reranker_name: str
    latency_ms: float | None = None
    input_chunk_count: int
    output_chunk_count: int
    reranked_chunks: list[RerankedChunk]


class ContextAssembly(BaseModel):
    assembled_context: str
    chunks_used: list[str] = Field(default_factory=list)
    total_tokens: int | None = None
    window_size: int | None = None
    truncation_applied: bool = False
    latency_ms: float | None = None


class LLMGenerationStep(BaseModel):
    model_name: str
    generated_answer: str
    latency_ms: float | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    system_prompt: str | None = None
    user_query: str | None = None
    full_prompt: str | None = None


class TraceError(BaseModel):
    step: str
    message: str
    error_type: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TracePrivacyConfig(BaseModel):
    store_full_prompt: bool = False
    store_full_embeddings: bool = False
    redacted_fields: list[str] = Field(default_factory=list)


class TraceSummary(BaseModel):
    total_latency_ms: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost_usd: float | None = None
    retrieved_chunk_count: int = 0
    context_chunk_count: int = 0
    validation_warnings: list[str] = Field(default_factory=list)


class RAGTrace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = "0.1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: TraceStatus = "pending"
    pipeline_name: str = "default"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    privacy: TracePrivacyConfig = Field(default_factory=TracePrivacyConfig)

    query_input: QueryInput | None = None
    embedding_step: EmbeddingStep | None = None
    retrieval_step: RetrievalStep | None = None
    reranker_step: RerankerStep | None = None
    context_assembly: ContextAssembly | None = None
    llm_generation: LLMGenerationStep | None = None

    final_answer: str | None = None
    errors: list[TraceError] = Field(default_factory=list)
    summary: TraceSummary = Field(default_factory=TraceSummary)

    quality_score: float | None = None
    issues_detected: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def finalize(self) -> "RAGTrace":
        total_latency = 0.0
        has_latency = False
        validation_warnings: list[str] = []
        for step in (
            self.embedding_step,
            self.retrieval_step,
            self.reranker_step,
            self.context_assembly,
            self.llm_generation,
        ):
            latency = getattr(step, "latency_ms", None) if step else None
            if latency is not None:
                total_latency += latency
                has_latency = True

        if self.retrieval_step and self.context_assembly:
            retrieved_chunk_ids = {chunk.chunk_id for chunk in self.retrieval_step.retrieved_chunks}
            missing_chunk_ids = [
                chunk_id for chunk_id in self.context_assembly.chunks_used if chunk_id not in retrieved_chunk_ids
            ]
            if missing_chunk_ids:
                validation_warnings.append(
                    "Context references chunk ids that were not recorded in retrieval: "
                    + ", ".join(missing_chunk_ids)
                )

        if self.reranker_step and self.retrieval_step:
            reranked_chunk_ids = {chunk.chunk_id for chunk in self.reranker_step.reranked_chunks}
            retrieved_chunk_ids = {chunk.chunk_id for chunk in self.retrieval_step.retrieved_chunks}
            unknown_chunk_ids = sorted(reranked_chunk_ids - retrieved_chunk_ids)
            if unknown_chunk_ids:
                validation_warnings.append(
                    "Reranker references chunk ids that were not recorded in retrieval: "
                    + ", ".join(unknown_chunk_ids)
                )

        self.summary = TraceSummary(
            total_latency_ms=round(total_latency, 3) if has_latency else None,
            total_input_tokens=self.llm_generation.tokens_input if self.llm_generation else None,
            total_output_tokens=self.llm_generation.tokens_output if self.llm_generation else None,
            total_cost_usd=self.llm_generation.cost_usd if self.llm_generation else None,
            retrieved_chunk_count=len(self.retrieval_step.retrieved_chunks) if self.retrieval_step else 0,
            context_chunk_count=len(self.context_assembly.chunks_used) if self.context_assembly else 0,
            validation_warnings=validation_warnings,
        )

        if not self.final_answer and self.llm_generation:
            self.final_answer = self.llm_generation.generated_answer

        if self.errors:
            self.status = "failed"
        elif self.status == "pending":
            self.status = "completed"
        return self

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RAGTrace":
        source = Path(path)
        return cls.model_validate_json(source.read_text(encoding="utf-8"))

    @classmethod
    def save_batch(cls, traces: list["RAGTrace"], path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(trace.model_dump_json() for trace in traces)
        target.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return target

    @classmethod
    def load_batch(cls, path: str | Path) -> list["RAGTrace"]:
        source = Path(path)
        traces: list[RAGTrace] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces.append(cls.model_validate_json(line))
        return traces

    def analyze(self, weights: "HealthWeights | None" = None) -> "TraceAnalysis":
        from rag_debugger.analysis.analyzer import TraceAnalyzer

        analyzer = TraceAnalyzer(weights=weights)
        return analyzer.analyze_trace(self)
