from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_debugger.analysis.explanations import explain_trace
from rag_debugger.analysis.team import build_team_report
from rag_debugger.core.trace import RAGTrace
from rag_debugger.utils.dashboard_server import serve_dashboard_forever

app = typer.Typer(help="Inspect and analyze RAG traces.")
console = Console()


def _load_trace(path: Path) -> RAGTrace:
    return RAGTrace.load(path)


def _load_traces(path: Path) -> list[RAGTrace]:
    if path.is_file():
        if path.suffix == ".jsonl":
            return RAGTrace.load_batch(path)
        return [RAGTrace.load(path)]
    traces: list[RAGTrace] = []
    for candidate in sorted(path.glob("*.json")):
        traces.append(RAGTrace.load(candidate))
    for candidate in sorted(path.glob("*.jsonl")):
        traces.extend(RAGTrace.load_batch(candidate))
    return traces


def _format_delta(left_value: float | int | None, right_value: float | int | None, *, scale: float = 1.0) -> str:
    if left_value is None or right_value is None:
        return "n/a"
    delta = (right_value - left_value) * scale
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"


def _render_html_report(trace: RAGTrace) -> str:
    analysis = trace.analyze()
    findings = "".join(f"<li>{html.escape(item)}</li>" for item in analysis.findings) or "<li>None</li>"
    recommendations = (
        "".join(f"<li>{html.escape(item)}</li>" for item in analysis.recommendations) or "<li>None</li>"
    )
    retrieval_rows = ""
    if trace.retrieval_step:
        retrieval_rows = "".join(
            (
                "<tr>"
                f"<td>{html.escape(chunk.chunk_id)}</td>"
                f"<td>{chunk.rank}</td>"
                f"<td>{chunk.similarity_score:.2f}</td>"
                f"<td>{html.escape(chunk.source)}</td>"
                "</tr>"
            )
            for chunk in trace.retrieval_step.retrieved_chunks
        )

    reranker_rows = ""
    if trace.reranker_step:
        reranker_rows = "".join(
            (
                "<tr>"
                f"<td>{html.escape(chunk.chunk_id)}</td>"
                f"<td>{chunk.original_rank or ''}</td>"
                f"<td>{chunk.reranked_rank}</td>"
                f"<td>{'' if chunk.reranker_score is None else f'{chunk.reranker_score:.2f}'}</td>"
                "</tr>"
            )
            for chunk in trace.reranker_step.reranked_chunks
        )

    validation_warnings = "".join(
        f"<li>{html.escape(item)}</li>" for item in trace.summary.validation_warnings
    ) or "<li>None</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAG Debugger Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 2rem auto;
      max-width: 980px;
      line-height: 1.5;
      color: #18212b;
      background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
      padding: 0 1rem 3rem;
    }}
    .card {{
      background: white;
      border-radius: 14px;
      padding: 1.25rem 1.5rem;
      box-shadow: 0 10px 30px rgba(19, 45, 77, 0.08);
      margin-bottom: 1rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
    }}
    .metric {{
      background: #f5f8fb;
      border-radius: 12px;
      padding: 0.9rem;
    }}
    .label {{
      color: #516173;
      font-size: 0.92rem;
    }}
    .value {{
      font-size: 1.5rem;
      font-weight: 700;
      margin-top: 0.2rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border-bottom: 1px solid #e6ebf1;
      text-align: left;
      padding: 0.65rem 0.4rem;
      vertical-align: top;
    }}
    h1, h2 {{
      margin-bottom: 0.6rem;
    }}
    code {{
      background: #eff4fa;
      padding: 0.1rem 0.35rem;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>RAG Debugger Report</h1>
    <p><strong>Trace:</strong> <code>{html.escape(trace.trace_id)}</code></p>
    <p><strong>Pipeline:</strong> {html.escape(trace.pipeline_name)}</p>
    <p><strong>Status:</strong> {html.escape(trace.status)}</p>
    <p><strong>Query:</strong> {html.escape(trace.query_input.text if trace.query_input else "n/a")}</p>
  </div>

  <div class="card">
    <h2>Scores</h2>
    <div class="grid">
      <div class="metric"><div class="label">Overall health</div><div class="value">{analysis.overall_health:.1f}</div></div>
      <div class="metric"><div class="label">Retrieval</div><div class="value">{analysis.retrieval.score:.2f}</div></div>
      <div class="metric"><div class="label">Context</div><div class="value">{analysis.context.score:.2f}</div></div>
      <div class="metric"><div class="label">Grounding</div><div class="value">{analysis.grounding.score:.2f}</div></div>
      <div class="metric"><div class="label">Performance</div><div class="value">{analysis.performance.score:.2f}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Findings</h2>
    <ul>{findings}</ul>
    <h2>Recommendations</h2>
    <ul>{recommendations}</ul>
    <h2>Validation warnings</h2>
    <ul>{validation_warnings}</ul>
  </div>

  <div class="card">
    <h2>Retrieved Chunks</h2>
    <table>
      <thead>
        <tr><th>Chunk</th><th>Rank</th><th>Similarity</th><th>Source</th></tr>
      </thead>
      <tbody>{retrieval_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Reranker</h2>
    <p><strong>Name:</strong> {html.escape(trace.reranker_step.reranker_name if trace.reranker_step else "n/a")}</p>
    <table>
      <thead>
        <tr><th>Chunk</th><th>Original rank</th><th>Reranked rank</th><th>Reranker score</th></tr>
      </thead>
      <tbody>{reranker_rows or '<tr><td colspan="4">No reranker step recorded.</td></tr>'}</tbody>
    </table>
  </div>
</body>
</html>
"""


def _render_dashboard_html(traces: list[RAGTrace], refresh_seconds: int | None = None) -> str:
    analyzed = [(trace, trace.analyze()) for trace in traces]
    rows = "".join(
        (
            "<tr>"
            f"<td>{html.escape(trace.trace_id[:12])}</td>"
            f"<td>{html.escape(trace.pipeline_name)}</td>"
            f"<td>{analysis.overall_health:.1f}</td>"
            f"<td>{analysis.retrieval.score:.2f}</td>"
            f"<td>{analysis.grounding.score:.2f}</td>"
            f"<td>{trace.summary.total_latency_ms or 0:.0f}</td>"
            "</tr>"
        )
        for trace, analysis in analyzed
    )
    averages = {
        "health": sum(item[1].overall_health for item in analyzed) / len(analyzed),
        "retrieval": sum(item[1].retrieval.score for item in analyzed) / len(analyzed),
        "grounding": sum(item[1].grounding.score for item in analyzed) / len(analyzed),
        "latency": sum((item[0].summary.total_latency_ms or 0) for item in analyzed) / len(analyzed),
    }
    refresh_meta = (
        f'<meta http-equiv="refresh" content="{refresh_seconds}">' if refresh_seconds and refresh_seconds > 0 else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAG Debugger Dashboard</title>
  {refresh_meta}
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; background: #f5f8fb; color: #18212b; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1rem; }}
    .card {{ background: white; border-radius: 14px; padding: 1rem 1.2rem; box-shadow: 0 10px 25px rgba(19,45,77,0.08); }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 14px; overflow: hidden; }}
    th, td {{ padding: 0.75rem; border-bottom: 1px solid #e7edf4; text-align: left; }}
    h1 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <h1>RAG Debugger Dashboard</h1>
  <div class="grid">
    <div class="card"><strong>Trace count</strong><div>{len(traces)}</div></div>
    <div class="card"><strong>Avg health</strong><div>{averages['health']:.1f}</div></div>
    <div class="card"><strong>Avg retrieval</strong><div>{averages['retrieval']:.2f}</div></div>
    <div class="card"><strong>Avg grounding</strong><div>{averages['grounding']:.2f}</div></div>
    <div class="card"><strong>Avg latency ms</strong><div>{averages['latency']:.0f}</div></div>
  </div>
  <table>
    <thead>
      <tr><th>Trace</th><th>Pipeline</th><th>Health</th><th>Retrieval</th><th>Grounding</th><th>Latency ms</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


@app.command()
def view(path: Path) -> None:
    """Pretty print a trace file."""
    trace = _load_trace(path)
    table = Table(title=f"Trace {trace.trace_id}")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("Status", trace.status)
    table.add_row("Pipeline", trace.pipeline_name)
    table.add_row("Query", trace.query_input.text if trace.query_input else "n/a")
    table.add_row("Retrieved chunks", str(trace.summary.retrieved_chunk_count))
    table.add_row("Context chunks", str(trace.summary.context_chunk_count))
    table.add_row("Latency (ms)", str(trace.summary.total_latency_ms))
    table.add_row("Answer", trace.final_answer or "n/a")
    console.print(table)

    if trace.summary.validation_warnings:
        console.print(
            Panel(
                "\n".join(f"- {item}" for item in trace.summary.validation_warnings),
                title="Validation warnings",
            )
        )

    if trace.retrieval_step:
        chunks = Table(title="Retrieved Chunks")
        chunks.add_column("Chunk")
        chunks.add_column("Rank")
        chunks.add_column("Similarity")
        chunks.add_column("Source")
        for chunk in trace.retrieval_step.retrieved_chunks:
            chunks.add_row(chunk.chunk_id, str(chunk.rank), f"{chunk.similarity_score:.2f}", chunk.source)
        console.print(chunks)

    if trace.reranker_step:
        reranked = Table(title=f"Reranker ({trace.reranker_step.reranker_name})")
        reranked.add_column("Chunk")
        reranked.add_column("Original")
        reranked.add_column("Reranked")
        reranked.add_column("Score")
        for chunk in trace.reranker_step.reranked_chunks:
            reranked.add_row(
                chunk.chunk_id,
                str(chunk.original_rank or ""),
                str(chunk.reranked_rank),
                "" if chunk.reranker_score is None else f"{chunk.reranker_score:.2f}",
            )
        console.print(reranked)


@app.command()
def analyze(path: Path) -> None:
    """Analyze a trace and print findings."""
    trace = _load_trace(path)
    analysis = trace.analyze()

    summary = Table(title="RAG Pipeline Analysis")
    summary.add_column("Metric")
    summary.add_column("Score")
    summary.add_row("Overall health", f"{analysis.overall_health:.1f} ({analysis.status})")
    summary.add_row("Retrieval", f"{analysis.retrieval.score:.2f}")
    summary.add_row("Context", f"{analysis.context.score:.2f}")
    summary.add_row("Grounding", f"{analysis.grounding.score:.2f}")
    summary.add_row("Performance", f"{analysis.performance.score:.2f}")
    console.print(summary)

    if analysis.findings:
        console.print(Panel("\n".join(f"- {item}" for item in analysis.findings), title="Findings"))
    if analysis.recommendations:
        console.print(Panel("\n".join(f"- {item}" for item in analysis.recommendations), title="Recommendations"))


@app.command()
def diff(left: Path, right: Path) -> None:
    """Compare two traces side by side."""
    left_trace = _load_trace(left)
    right_trace = _load_trace(right)
    left_analysis = left_trace.analyze()
    right_analysis = right_trace.analyze()

    table = Table(title="Trace Diff")
    table.add_column("Metric")
    table.add_column("Left")
    table.add_column("Right")
    table.add_column("Delta")
    table.add_row("Trace", left_trace.trace_id[:12], right_trace.trace_id[:12], "")
    table.add_row("Overall health", f"{left_analysis.overall_health:.1f}", f"{right_analysis.overall_health:.1f}", _format_delta(left_analysis.overall_health, right_analysis.overall_health))
    table.add_row("Retrieval", f"{left_analysis.retrieval.score:.2f}", f"{right_analysis.retrieval.score:.2f}", _format_delta(left_analysis.retrieval.score, right_analysis.retrieval.score))
    table.add_row("Context", f"{left_analysis.context.score:.2f}", f"{right_analysis.context.score:.2f}", _format_delta(left_analysis.context.score, right_analysis.context.score))
    table.add_row("Grounding", f"{left_analysis.grounding.score:.2f}", f"{right_analysis.grounding.score:.2f}", _format_delta(left_analysis.grounding.score, right_analysis.grounding.score))
    table.add_row("Performance", f"{left_analysis.performance.score:.2f}", f"{right_analysis.performance.score:.2f}", _format_delta(left_analysis.performance.score, right_analysis.performance.score))
    table.add_row("Latency (ms)", f"{left_trace.summary.total_latency_ms or 0:.0f}", f"{right_trace.summary.total_latency_ms or 0:.0f}", _format_delta(left_trace.summary.total_latency_ms, right_trace.summary.total_latency_ms))
    table.add_row("Retrieved chunks", str(left_trace.summary.retrieved_chunk_count), str(right_trace.summary.retrieved_chunk_count), _format_delta(left_trace.summary.retrieved_chunk_count, right_trace.summary.retrieved_chunk_count))
    table.add_row("Context chunks", str(left_trace.summary.context_chunk_count), str(right_trace.summary.context_chunk_count), _format_delta(left_trace.summary.context_chunk_count, right_trace.summary.context_chunk_count))
    table.add_row("Reranker", left_trace.reranker_step.reranker_name if left_trace.reranker_step else "n/a", right_trace.reranker_step.reranker_name if right_trace.reranker_step else "n/a", "")
    console.print(table)

    if left_trace.query_input or right_trace.query_input:
        console.print(
            Panel(
                f"Left query: {left_trace.query_input.text if left_trace.query_input else 'n/a'}\n"
                f"Right query: {right_trace.query_input.text if right_trace.query_input else 'n/a'}",
                title="Queries",
            )
        )

    console.print(
        Panel(
            "Left:\n"
            + "\n".join(f"- {item}" for item in left_analysis.recommendations)
            + "\n\nRight:\n"
            + "\n".join(f"- {item}" for item in right_analysis.recommendations),
            title="Recommendations",
        )
    )


@app.command()
def stats(path: Path) -> None:
    """Aggregate metrics across a directory of traces."""
    traces = _load_traces(path)
    if not traces:
        raise typer.BadParameter("No trace files were found at the provided path.")

    table = Table(title=f"Trace Stats ({len(traces)} traces)")
    table.add_column("Trace")
    table.add_column("Health")
    table.add_column("Retrieval")
    table.add_column("Grounding")
    table.add_column("Latency (ms)")

    health_values: list[float] = []
    retrieval_values: list[float] = []
    grounding_values: list[float] = []
    latency_values: list[float] = []

    for trace in traces:
        analysis = trace.analyze()
        health_values.append(analysis.overall_health)
        retrieval_values.append(analysis.retrieval.score * 100)
        grounding_values.append(analysis.grounding.score * 100)
        if trace.summary.total_latency_ms is not None:
            latency_values.append(trace.summary.total_latency_ms)
        table.add_row(
            trace.trace_id[:8],
            f"{analysis.overall_health:.1f}",
            f"{analysis.retrieval.score * 100:.1f}",
            f"{analysis.grounding.score * 100:.1f}",
            f"{trace.summary.total_latency_ms or 0:.0f}",
        )

    if traces:
        table.add_section()
        table.add_row(
            "Average",
            f"{sum(health_values) / len(health_values):.1f}",
            f"{sum(retrieval_values) / len(retrieval_values):.1f}",
            f"{sum(grounding_values) / len(grounding_values):.1f}",
            f"{(sum(latency_values) / len(latency_values)) if latency_values else 0:.0f}",
        )

    console.print(table)


@app.command()
def aggregate(path: Path, group_by: str = typer.Option("pipeline", "--group-by")) -> None:
    """Aggregate many traces by pipeline, status, or tag."""
    traces = _load_traces(path)
    if not traces:
        raise typer.BadParameter("No trace files were found at the provided path.")
    if group_by not in {"pipeline", "status", "tag"}:
        raise typer.BadParameter("group_by must be one of: pipeline, status, tag")

    grouped: dict[str, list[RAGTrace]] = {}
    for trace in traces:
        if group_by == "pipeline":
            keys = [trace.pipeline_name]
        elif group_by == "status":
            keys = [trace.status]
        else:
            keys = trace.tags or ["untagged"]
        for key in keys:
            grouped.setdefault(key, []).append(trace)

    table = Table(title=f"Aggregated Trace Stats by {group_by}")
    table.add_column(group_by.capitalize())
    table.add_column("Count")
    table.add_column("Avg health")
    table.add_column("Avg retrieval")
    table.add_column("Avg grounding")
    table.add_column("Avg latency")
    for key, group in sorted(grouped.items()):
        analyses = [trace.analyze() for trace in group]
        avg_health = sum(item.overall_health for item in analyses) / len(analyses)
        avg_retrieval = sum(item.retrieval.score for item in analyses) / len(analyses)
        avg_grounding = sum(item.grounding.score for item in analyses) / len(analyses)
        avg_latency = sum((trace.summary.total_latency_ms or 0) for trace in group) / len(group)
        table.add_row(key, str(len(group)), f"{avg_health:.1f}", f"{avg_retrieval:.2f}", f"{avg_grounding:.2f}", f"{avg_latency:.0f}")
    console.print(table)


@app.command()
def dashboard(path: Path, output: Path = typer.Option(..., "--output")) -> None:
    """Generate a static HTML dashboard for a directory of traces."""
    traces = _load_traces(path)
    if not traces:
        raise typer.BadParameter("No trace files were found at the provided path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_dashboard_html(traces), encoding="utf-8")
    console.print(f"Wrote dashboard to {output}")


@app.command()
def serve_dashboard(
    path: Path,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    refresh_seconds: int = typer.Option(5, "--refresh-seconds"),
) -> None:
    """Serve a live auto-refreshing dashboard."""
    traces = _load_traces(path)
    if not traces:
        raise typer.BadParameter("No trace files were found at the provided path.")
    console.print(f"Serving live dashboard at http://{host}:{port}")
    try:
        serve_dashboard_forever(
            lambda: _load_traces(path),
            _render_dashboard_html,
            host=host,
            port=port,
            refresh_seconds=refresh_seconds,
        )
    except KeyboardInterrupt:
        console.print("Stopped live dashboard.")


@app.command()
def team_report(
    path: Path,
    group_by: str = typer.Option("pipeline", "--group-by"),
    format: str = typer.Option("table", "--format"),
) -> None:
    """Build a richer team analytics report."""
    traces = _load_traces(path)
    if not traces:
        raise typer.BadParameter("No trace files were found at the provided path.")
    report = build_team_report(traces, group_by=group_by)

    if format == "json":
        console.print_json(json.dumps(report.model_dump(mode="json"), indent=2))
        return
    if format != "table":
        raise typer.BadParameter("format must be one of: table, json")

    summary = Table(title="Team Report")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total traces", str(report.total_traces))
    summary.add_row("Status counts", ", ".join(f"{key}={value}" for key, value in sorted(report.status_counts.items())))
    summary.add_row("Tag counts", ", ".join(f"{key}={value}" for key, value in sorted(report.tag_counts.items())))
    console.print(summary)

    groups = Table(title=f"Grouped Metrics by {group_by}")
    groups.add_column("Key")
    groups.add_column("Count")
    groups.add_column("Avg health")
    groups.add_column("Avg retrieval")
    groups.add_column("Avg grounding")
    groups.add_column("Avg latency")
    for item in report.grouped_metrics:
        groups.add_row(
            item.key,
            str(item.count),
            f"{item.avg_health:.1f}",
            f"{item.avg_retrieval:.2f}",
            f"{item.avg_grounding:.2f}",
            f"{item.avg_latency_ms:.0f}",
        )
    console.print(groups)

    if report.top_findings:
        findings = Panel("\n".join(f"- {item.finding} ({item.count})" for item in report.top_findings), title="Top Findings")
        console.print(findings)


@app.command()
def explain(path: Path, style: str = typer.Option("heuristic", "--style")) -> None:
    """Generate an explanation summary for a trace."""
    trace = _load_trace(path)
    analysis = trace.analyze()
    if style not in {"heuristic", "structured"}:
        raise typer.BadParameter("style must be one of: heuristic, structured")
    console.print(Panel(explain_trace(trace, analysis, style=style), title="Explanation"))


@app.command()
def export(
    path: Path,
    format: str = typer.Option("json", "--format"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Export trace data as json, csv, html, or a pretty summary."""
    trace = _load_trace(path)
    analysis = trace.analyze()

    if format == "json":
        payload = json.dumps(trace.model_dump(mode="json"), indent=2)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            console.print(f"Wrote JSON export to {output}")
            return
        console.print_json(payload)
        return

    if format == "pretty":
        analyze(path)
        return

    if format == "csv":
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open("w", encoding="utf-8", newline="")
            close_handle = True
        else:
            handle = typer.get_text_stream("stdout")
            close_handle = False
        try:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "trace_id",
                    "pipeline_name",
                    "status",
                    "overall_health",
                    "retrieval_score",
                    "grounding_score",
                    "latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "trace_id": trace.trace_id,
                    "pipeline_name": trace.pipeline_name,
                    "status": trace.status,
                    "overall_health": analysis.overall_health,
                    "retrieval_score": analysis.retrieval.score,
                    "grounding_score": analysis.grounding.score,
                    "latency_ms": trace.summary.total_latency_ms,
                }
            )
        finally:
            if close_handle:
                handle.close()
        if output:
            console.print(f"Wrote CSV export to {output}")
        return

    if format == "html":
        report = _render_html_report(trace)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report, encoding="utf-8")
            console.print(f"Wrote HTML report to {output}")
            return
        console.print(report)
        return

    raise typer.BadParameter("format must be one of: json, csv, html, pretty")
