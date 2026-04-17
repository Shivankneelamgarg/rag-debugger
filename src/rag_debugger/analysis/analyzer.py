from __future__ import annotations

import re
import statistics
from typing import Literal

from pydantic import BaseModel, Field

from rag_debugger.analysis.metrics import clamp, lexical_overlap, rank_score
from rag_debugger.core.trace import RAGTrace, RetrievedChunk


class HealthWeights(BaseModel):
    retrieval: float = 40.0
    context: float = 25.0
    grounding: float = 25.0
    performance: float = 10.0

    @property
    def total(self) -> float:
        return self.retrieval + self.context + self.grounding + self.performance


class ChunkAnalysis(BaseModel):
    chunk_id: str
    rank: int
    similarity: float
    quality: Literal["excellent", "good", "medium", "poor"]
    in_context: bool
    redundant_with: list[str] = Field(default_factory=list)
    recommendation: str


class RetrievalAnalysis(BaseModel):
    score: float
    average_similarity: float
    chunks: list[ChunkAnalysis]
    findings: list[str] = Field(default_factory=list)


class ContextAnalysis(BaseModel):
    score: float
    truncation_applied: bool
    context_pressure: float | None = None
    utilization_rate: float | None = None
    findings: list[str] = Field(default_factory=list)


class GroundingAnalysis(BaseModel):
    score: float
    grounded_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    findings: list[str] = Field(default_factory=list)


class PerformanceAnalysis(BaseModel):
    score: float
    total_latency_ms: float | None = None
    total_cost_usd: float | None = None
    findings: list[str] = Field(default_factory=list)


class TraceAnalysis(BaseModel):
    overall_health: float
    status: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"]
    retrieval: RetrievalAnalysis
    context: ContextAnalysis
    grounding: GroundingAnalysis
    performance: PerformanceAnalysis
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    weights: HealthWeights


class TraceAnalyzer:
    def __init__(self, *, weights: HealthWeights | None = None) -> None:
        self.weights = weights or HealthWeights()

    def analyze_trace(self, trace: RAGTrace) -> TraceAnalysis:
        retrieval = self._analyze_retrieval(trace)
        context = self._analyze_context(trace)
        grounding = self._analyze_grounding(trace)
        performance = self._analyze_performance(trace)

        total_weight = self.weights.total or 1.0
        overall = (
            retrieval.score * self.weights.retrieval
            + context.score * self.weights.context
            + grounding.score * self.weights.grounding
            + performance.score * self.weights.performance
        ) / total_weight
        overall *= 100

        status: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"]
        if overall >= 90:
            status = "EXCELLENT"
        elif overall >= 70:
            status = "GOOD"
        elif overall >= 50:
            status = "FAIR"
        else:
            status = "POOR"

        findings = retrieval.findings + context.findings + grounding.findings + performance.findings
        recommendations = self._build_recommendations(trace, retrieval, context, grounding, performance)

        trace.quality_score = round(overall, 2)
        trace.issues_detected = findings
        trace.recommendations = recommendations

        return TraceAnalysis(
            overall_health=round(overall, 2),
            status=status,
            retrieval=retrieval,
            context=context,
            grounding=grounding,
            performance=performance,
            findings=findings,
            recommendations=recommendations,
            weights=self.weights,
        )

    def _analyze_retrieval(self, trace: RAGTrace) -> RetrievalAnalysis:
        retrieval_step = trace.retrieval_step
        context_step = trace.context_assembly
        if not retrieval_step:
            return RetrievalAnalysis(
                score=0.5,
                average_similarity=0.0,
                chunks=[],
                findings=["No retrieval step was recorded."],
            )

        chunks = retrieval_step.retrieved_chunks
        similarities = [chunk.similarity_score for chunk in chunks]
        average_similarity = statistics.mean(similarities) if similarities else 0.0
        chunk_analyses: list[ChunkAnalysis] = []
        findings: list[str] = []
        context_ids = set(context_step.chunks_used if context_step else [])
        excluded_high_quality = 0

        for chunk in chunks:
            quality = self._quality_label(chunk.similarity_score)
            in_context = chunk.chunk_id in context_ids
            redundant_with = self._find_redundancy(chunk, chunks)

            if chunk.similarity_score >= 0.85 and not in_context:
                excluded_high_quality += 1

            if chunk.similarity_score < 0.7 and in_context:
                recommendation = "Low-similarity chunk reached final context."
                findings.append(f"Chunk {chunk.chunk_id} has low similarity but was included in context.")
            elif redundant_with:
                recommendation = "Chunk appears redundant with nearby retrieved chunks."
                findings.append(f"Chunk {chunk.chunk_id} may be redundant with {', '.join(redundant_with)}.")
            elif chunk.similarity_score >= 0.9:
                recommendation = "Strong retrieval result."
            else:
                recommendation = "Acceptable retrieval result."

            chunk_analyses.append(
                ChunkAnalysis(
                    chunk_id=chunk.chunk_id,
                    rank=chunk.rank,
                    similarity=round(chunk.similarity_score, 3),
                    quality=quality,
                    in_context=in_context,
                    redundant_with=redundant_with,
                    recommendation=recommendation,
                )
            )

        rank_weighted_scores = [
            clamp((chunk.similarity_score * 0.75) + (rank_score(chunk.rank) * 0.25))
            for chunk in chunks
        ]
        score = statistics.mean(rank_weighted_scores) if rank_weighted_scores else 0.0

        if excluded_high_quality:
            findings.append(f"{excluded_high_quality} high-quality chunk(s) were retrieved but not used in context.")
        if average_similarity < 0.75:
            findings.append(f"Average retrieval similarity is low at {average_similarity:.2f}.")

        return RetrievalAnalysis(
            score=round(score, 3),
            average_similarity=round(average_similarity, 3),
            chunks=chunk_analyses,
            findings=findings,
        )

    def _analyze_context(self, trace: RAGTrace) -> ContextAnalysis:
        retrieval_step = trace.retrieval_step
        context_step = trace.context_assembly
        if not context_step:
            return ContextAnalysis(
                score=0.5,
                truncation_applied=False,
                findings=["No context assembly step was recorded."],
            )

        findings: list[str] = []
        utilization_rate: float | None = None
        if retrieval_step and retrieval_step.num_chunks_retrieved:
            utilization_rate = len(context_step.chunks_used) / retrieval_step.num_chunks_retrieved

        context_pressure: float | None = None
        if context_step.window_size and context_step.total_tokens is not None:
            context_pressure = clamp(context_step.total_tokens / context_step.window_size)

        score = 1.0
        if context_step.truncation_applied:
            findings.append("Context truncation was applied.")
            score -= 0.3
        if context_pressure is not None and context_pressure > 0.9:
            findings.append("Context window is close to full capacity.")
            score -= 0.2
        if utilization_rate is not None and utilization_rate < 0.5:
            findings.append("Less than half of retrieved chunks were used in final context.")
            score -= 0.15

        return ContextAnalysis(
            score=round(clamp(score), 3),
            truncation_applied=context_step.truncation_applied,
            context_pressure=round(context_pressure, 3) if context_pressure is not None else None,
            utilization_rate=round(utilization_rate, 3) if utilization_rate is not None else None,
            findings=findings,
        )

    def _analyze_grounding(self, trace: RAGTrace) -> GroundingAnalysis:
        answer = trace.final_answer or ""
        context = trace.context_assembly.assembled_context if trace.context_assembly else ""
        if not answer:
            return GroundingAnalysis(
                score=0.5,
                confidence="LOW",
                findings=["No final answer was recorded."],
            )
        if not context:
            return GroundingAnalysis(
                score=0.4,
                confidence="LOW",
                unsupported_claims=[answer],
                findings=["No context is available to ground the answer."],
            )

        claims = [segment.strip() for segment in answer.split(".") if segment.strip()]
        context_segments = [segment.strip() for segment in context.split(".") if segment.strip()]
        grounded: list[str] = []
        unsupported: list[str] = []
        for claim in claims:
            comparisons = [lexical_overlap(claim, context)]
            comparisons.extend(lexical_overlap(claim, segment) for segment in context_segments)
            if max(comparisons) >= 0.15 and self._claim_has_numeric_support(claim, context, context_segments):
                grounded.append(claim)
            else:
                unsupported.append(claim)

        total_claims = max(len(claims), 1)
        score = len(grounded) / total_claims
        if len(claims) >= 4:
            confidence = "HIGH"
        elif len(claims) >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        findings: list[str] = []
        if unsupported:
            findings.append(f"{len(unsupported)} answer claim(s) are weakly supported by the retrieved context.")

        return GroundingAnalysis(
            score=round(score, 3),
            grounded_claims=grounded,
            unsupported_claims=unsupported,
            confidence=confidence,
            findings=findings,
        )

    def _analyze_performance(self, trace: RAGTrace) -> PerformanceAnalysis:
        total_latency = trace.summary.total_latency_ms
        total_cost = trace.summary.total_cost_usd
        score = 1.0
        findings: list[str] = []

        if total_latency is not None:
            if total_latency > 3000:
                score -= 0.45
                findings.append(f"Total latency is high at {total_latency:.0f} ms.")
            elif total_latency > 1500:
                score -= 0.2
                findings.append(f"Total latency is moderate at {total_latency:.0f} ms.")

        total_tokens = (trace.summary.total_input_tokens or 0) + (trace.summary.total_output_tokens or 0)
        if total_tokens > 6000:
            score -= 0.2
            findings.append(f"Token usage is high at {total_tokens} tokens.")
        elif total_tokens > 3000:
            score -= 0.1
            findings.append(f"Token usage is moderate at {total_tokens} tokens.")

        if total_cost is not None and total_cost > 0.05:
            score -= 0.15
            findings.append(f"Per-query cost is high at ${total_cost:.4f}.")

        return PerformanceAnalysis(
            score=round(clamp(score), 3),
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            findings=findings,
        )

    def _find_redundancy(self, chunk: RetrievedChunk, chunks: list[RetrievedChunk]) -> list[str]:
        redundant_with: list[str] = []
        for other in chunks:
            if other.chunk_id == chunk.chunk_id:
                continue
            if lexical_overlap(chunk.text, other.text) >= 0.8:
                redundant_with.append(other.chunk_id)
        return sorted(set(redundant_with))

    def _quality_label(self, similarity: float) -> Literal["excellent", "good", "medium", "poor"]:
        if similarity >= 0.9:
            return "excellent"
        if similarity >= 0.8:
            return "good"
        if similarity >= 0.7:
            return "medium"
        return "poor"

    def _build_recommendations(
        self,
        trace: RAGTrace,
        retrieval: RetrievalAnalysis,
        context: ContextAnalysis,
        grounding: GroundingAnalysis,
        performance: PerformanceAnalysis,
    ) -> list[str]:
        recommendations: list[str] = []
        if retrieval.average_similarity < 0.75:
            recommendations.append("Improve retrieval quality with better embeddings, chunking, or search parameters.")
        if any(chunk.in_context and chunk.similarity < 0.7 for chunk in retrieval.chunks):
            recommendations.append("Filter low-similarity chunks before building the final context.")
        if context.truncation_applied:
            recommendations.append("Reduce retrieved context size or move to a model with a larger context window.")
        if context.context_pressure is not None and context.context_pressure > 0.9:
            recommendations.append("Context is near the token limit. Summarize or rerank chunks before prompting.")
        if grounding.unsupported_claims:
            recommendations.append("Strengthen prompt grounding instructions and verify unsupported answer claims.")
        if performance.total_latency_ms and performance.total_latency_ms > 1500:
            recommendations.append("Latency is elevated. Consider caching embeddings or retrieval results.")
        if not recommendations:
            recommendations.append("Retrieval quality is strong. Keep current search settings.")
            recommendations.append("Grounding looks healthy overall.")
        return recommendations

    def _claim_has_numeric_support(self, claim: str, context: str, context_segments: list[str]) -> bool:
        claim_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
        if not claim_numbers:
            return True

        searchable_segments = [context, *context_segments]
        for number in claim_numbers:
            if not any(number in segment for segment in searchable_segments):
                return False
        return True
