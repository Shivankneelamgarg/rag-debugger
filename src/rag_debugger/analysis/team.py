from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

from pydantic import BaseModel, Field

from rag_debugger.core.trace import RAGTrace


class AnalyticsGroup(BaseModel):
    key: str
    count: int
    avg_health: float
    avg_retrieval: float
    avg_grounding: float
    avg_latency_ms: float


class DailyTrendPoint(BaseModel):
    day: str
    count: int
    avg_health: float


class FindingFrequency(BaseModel):
    finding: str
    count: int


class TeamAnalyticsReport(BaseModel):
    total_traces: int
    grouped_metrics: list[AnalyticsGroup]
    status_counts: dict[str, int]
    tag_counts: dict[str, int]
    daily_health_trend: list[DailyTrendPoint]
    top_findings: list[FindingFrequency]


def build_team_report(traces: list[RAGTrace], *, group_by: str = "pipeline") -> TeamAnalyticsReport:
    if group_by not in {"pipeline", "status", "tag"}:
        raise ValueError("group_by must be one of: pipeline, status, tag")

    grouped: dict[str, list[RAGTrace]] = defaultdict(list)
    finding_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    daily_traces: dict[date, list[RAGTrace]] = defaultdict(list)

    for trace in traces:
        analysis = trace.analyze()
        status_counter[trace.status] += 1
        trace_tags = trace.tags or ["untagged"]
        for tag in trace_tags:
            tag_counter[tag] += 1

        if group_by == "pipeline":
            group_keys = [trace.pipeline_name]
        elif group_by == "status":
            group_keys = [trace.status]
        else:
            group_keys = trace_tags

        for key in group_keys:
            grouped[key].append(trace)

        for finding in analysis.findings:
            finding_counter[finding] += 1

        daily_traces[trace.timestamp.date()].append(trace)

    grouped_metrics: list[AnalyticsGroup] = []
    for key, group in sorted(grouped.items()):
        analyses = [trace.analyze() for trace in group]
        grouped_metrics.append(
            AnalyticsGroup(
                key=key,
                count=len(group),
                avg_health=round(sum(item.overall_health for item in analyses) / len(analyses), 2),
                avg_retrieval=round(sum(item.retrieval.score for item in analyses) / len(analyses), 3),
                avg_grounding=round(sum(item.grounding.score for item in analyses) / len(analyses), 3),
                avg_latency_ms=round(
                    sum((trace.summary.total_latency_ms or 0.0) for trace in group) / len(group), 2
                ),
            )
        )

    daily_health_trend: list[DailyTrendPoint] = []
    for day, group in sorted(daily_traces.items()):
        analyses = [trace.analyze() for trace in group]
        daily_health_trend.append(
            DailyTrendPoint(
                day=day.isoformat(),
                count=len(group),
                avg_health=round(sum(item.overall_health for item in analyses) / len(analyses), 2),
            )
        )

    top_findings = [
        FindingFrequency(finding=finding, count=count) for finding, count in finding_counter.most_common(10)
    ]

    return TeamAnalyticsReport(
        total_traces=len(traces),
        grouped_metrics=grouped_metrics,
        status_counts=dict(status_counter),
        tag_counts=dict(tag_counter),
        daily_health_trend=daily_health_trend,
        top_findings=top_findings,
    )
