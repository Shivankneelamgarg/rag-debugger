"""Public package interface for RAG Debugger."""

from rag_debugger.analysis.analyzer import HealthWeights, TraceAnalysis
from rag_debugger.analysis.explanations import (
    ExternalExplanationProvider,
    HeuristicExplanationProvider,
    StructuredExplanationProvider,
    explain_trace,
)
from rag_debugger.analysis.team import TeamAnalyticsReport, build_team_report
from rag_debugger.core.trace import RAGTrace
from rag_debugger.core.tracer import RAGTracer
from rag_debugger.integrations.auto import AutoInstrumentedRAGPipeline
from rag_debugger.integrations.langchain import LangChainTraceAdapter
from rag_debugger.integrations.llamaindex import LlamaIndexTraceAdapter

__all__ = [
    "AutoInstrumentedRAGPipeline",
    "ExternalExplanationProvider",
    "HealthWeights",
    "HeuristicExplanationProvider",
    "LangChainTraceAdapter",
    "LlamaIndexTraceAdapter",
    "RAGTrace",
    "RAGTracer",
    "StructuredExplanationProvider",
    "TeamAnalyticsReport",
    "TraceAnalysis",
    "build_team_report",
    "explain_trace",
]
