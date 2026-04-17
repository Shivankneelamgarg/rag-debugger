from typer.testing import CliRunner

from rag_debugger.cli.commands import app


runner = CliRunner()


def test_cli_view_command() -> None:
    result = runner.invoke(app, ["view", "examples/healthy_trace.json"])
    assert result.exit_code == 0
    assert "Retrieved Chunks" in result.stdout


def test_cli_analyze_command() -> None:
    result = runner.invoke(app, ["analyze", "examples/healthy_trace.json"])
    assert result.exit_code == 0
    assert "RAG Pipeline Analysis" in result.stdout
    assert "Recommendations" in result.stdout


def test_cli_stats_command() -> None:
    result = runner.invoke(app, ["stats", "examples"])
    assert result.exit_code == 0
    assert "Average" in result.stdout


def test_cli_export_csv() -> None:
    result = runner.invoke(app, ["export", "examples/healthy_trace.json", "--format", "csv"])
    assert result.exit_code == 0
    assert "trace_id,pipeline_name,status" in result.stdout


def test_cli_diff_command() -> None:
    result = runner.invoke(app, ["diff", "examples/healthy_trace.json", "examples/low_retrieval_trace.json"])
    assert result.exit_code == 0
    assert "Trace Diff" in result.stdout
    assert "Overall health" in result.stdout


def test_cli_export_html(tmp_path) -> None:
    output = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["export", "examples/healthy_trace.json", "--format", "html", "--output", str(output)],
    )
    assert result.exit_code == 0
    assert output.exists()
    assert "RAG Debugger Report" in output.read_text(encoding="utf-8")


def test_cli_aggregate_command() -> None:
    result = runner.invoke(app, ["aggregate", "examples", "--group-by", "pipeline"])
    assert result.exit_code == 0
    assert "Aggregated Trace Stats by pipeline" in result.stdout


def test_cli_dashboard_command(tmp_path) -> None:
    output = tmp_path / "dashboard.html"
    result = runner.invoke(app, ["dashboard", "examples", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    assert "RAG Debugger Dashboard" in output.read_text(encoding="utf-8")


def test_cli_explain_command() -> None:
    result = runner.invoke(app, ["explain", "examples/healthy_trace.json"])
    assert result.exit_code == 0
    assert "Explanation" in result.stdout


def test_cli_explain_structured_command() -> None:
    result = runner.invoke(app, ["explain", "examples/healthy_trace.json", "--style", "structured"])
    assert result.exit_code == 0
    assert "Score breakdown" in result.stdout


def test_cli_team_report_command() -> None:
    result = runner.invoke(app, ["team-report", "examples", "--group-by", "tag"])
    assert result.exit_code == 0
    assert "Team Report" in result.stdout
    assert "Grouped Metrics by tag" in result.stdout
