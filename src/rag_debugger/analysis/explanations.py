from __future__ import annotations

from typing import Callable, Protocol

from rag_debugger.analysis.analyzer import TraceAnalysis
from rag_debugger.core.trace import RAGTrace


class ExplanationProvider(Protocol):
    def generate(self, trace: RAGTrace, analysis: TraceAnalysis) -> str: ...


ExplanationGenerator = Callable[[RAGTrace, TraceAnalysis], str]


class HeuristicExplanationProvider:
    def generate(self, trace: RAGTrace, analysis: TraceAnalysis) -> str:
        parts: list[str] = []
        query = trace.query_input.text if trace.query_input else "unknown query"
        parts.append(f"Query: {query}")
        parts.append(f"Overall health: {analysis.overall_health:.1f} ({analysis.status})")

        if analysis.findings:
            parts.append("Main findings:")
            parts.extend(f"- {item}" for item in analysis.findings[:5])
        else:
            parts.append("Main findings: no major issues detected.")

        if analysis.recommendations:
            parts.append("Recommended next actions:")
            parts.extend(f"- {item}" for item in analysis.recommendations[:5])

        return "\n".join(parts)


class StructuredExplanationProvider:
    def generate(self, trace: RAGTrace, analysis: TraceAnalysis) -> str:
        query = trace.query_input.text if trace.query_input else "unknown query"
        sections = [
            f"Query: {query}",
            f"Status: {trace.status}",
            f"Overall health: {analysis.overall_health:.1f} ({analysis.status})",
            "",
            "Score breakdown:",
            f"- Retrieval: {analysis.retrieval.score:.2f}",
            f"- Context: {analysis.context.score:.2f}",
            f"- Grounding: {analysis.grounding.score:.2f}",
            f"- Performance: {analysis.performance.score:.2f}",
            "",
            "Primary findings:",
        ]
        if analysis.findings:
            sections.extend(f"- {item}" for item in analysis.findings[:8])
        else:
            sections.append("- No major issues detected.")

        sections.append("")
        sections.append("Recommended actions:")
        if analysis.recommendations:
            sections.extend(f"- {item}" for item in analysis.recommendations[:8])
        else:
            sections.append("- No immediate action required.")

        return "\n".join(sections)


class ExternalExplanationProvider:
    def __init__(self, generator: ExplanationGenerator) -> None:
        self.generator = generator

    def generate(self, trace: RAGTrace, analysis: TraceAnalysis) -> str:
        return self.generator(trace, analysis)


def explain_trace(
    trace: RAGTrace,
    analysis: TraceAnalysis | None = None,
    *,
    style: str = "heuristic",
    provider: ExplanationProvider | ExplanationGenerator | None = None,
) -> str:
    resolved_analysis = analysis or trace.analyze()

    if provider is not None:
        if callable(provider) and not hasattr(provider, "generate"):
            return provider(trace, resolved_analysis)
        return provider.generate(trace, resolved_analysis)

    if style == "structured":
        return StructuredExplanationProvider().generate(trace, resolved_analysis)
    return HeuristicExplanationProvider().generate(trace, resolved_analysis)
