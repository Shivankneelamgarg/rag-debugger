from pathlib import Path

from examples.example_app import run_demo


def test_example_app_generates_trace(tmp_path: Path) -> None:
    path = run_demo("When was the company founded?", tmp_path)
    assert path.exists()
    contents = path.read_text(encoding="utf-8")
    assert '"pipeline_name": "example-demo"' in contents
    assert '"query_input"' in contents
