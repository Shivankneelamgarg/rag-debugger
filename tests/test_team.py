from urllib.request import urlopen

from rag_debugger.analysis.team import build_team_report
from rag_debugger.core.trace import RAGTrace
from rag_debugger.utils.dashboard_server import serve_dashboard_in_background
from tests.test_trace import make_trace


def test_build_team_report_group_by_tag() -> None:
    first = make_trace()
    first.tags = ["production", "healthy"]
    second = make_trace()
    second.pipeline_name = "other"
    second.tags = ["production"]
    report = build_team_report([first, second], group_by="tag")
    assert report.total_traces == 2
    keys = {item.key for item in report.grouped_metrics}
    assert "production" in keys
    assert report.status_counts["completed"] == 2


def test_live_dashboard_server_serves_html() -> None:
    traces = [make_trace()]
    server, thread, port = serve_dashboard_in_background(lambda: traces, lambda traces, refresh: "<html>live</html>")
    try:
        response = urlopen(f"http://127.0.0.1:{port}")
        body = response.read().decode("utf-8")
        assert "live" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
