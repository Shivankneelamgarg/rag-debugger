from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ASSETS = PROJECT_ROOT / "docs" / "assets"
EXAMPLE_ASSETS = PROJECT_ROOT / "examples" / "generated" / "assets"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_debugger import RAGTrace, build_team_report  # noqa: E402


def load_font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if mono:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Menlo.ttc",
                "/System/Library/Fonts/SFNSMono.ttf",
                "/Library/Fonts/Courier New.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/System/Library/Fonts/Supplemental/Verdana.ttf",
            ]
        )

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE_FONT = load_font(42)
H2_FONT = load_font(28)
BODY_FONT = load_font(22)
SMALL_FONT = load_font(18)
MONO_FONT = load_font(20, mono=True)
MONO_SMALL_FONT = load_font(18, mono=True)


def run_command(args: Sequence[str]) -> str:
    return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip("\n")


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, radius: int = 26, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def create_canvas(size: tuple[int, int] = (1600, 980), background: str = "#edf3fb") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 120), fill="#dfeaf8")
    draw.ellipse((50, 42, 66, 58), fill="#ff5f57")
    draw.ellipse((76, 42, 92, 58), fill="#febc2e")
    draw.ellipse((102, 42, 118, 58), fill="#28c840")
    return image, draw


def save_terminal_capture(command: str, output: str, destination: Path) -> Image.Image:
    image = Image.new("RGB", (1600, 980), "#0b1220")
    draw = ImageDraw.Draw(image)
    rounded(draw, (30, 30, 1570, 950), fill="#111b2d", outline="#1c2a45", radius=28, width=2)
    draw.text((70, 70), "CLI Analysis", font=TITLE_FONT, fill="#f7fbff")
    rounded(draw, (70, 135, 1530, 188), fill="#0a223b", radius=14)
    draw.text((90, 148), f"$ {command}", font=MONO_FONT, fill="#a3f7bf")

    y = 225
    for line in output.splitlines():
        draw.text((90, y), line, font=MONO_SMALL_FONT, fill="#dce6f7")
        bbox = draw.textbbox((90, y), line or " ", font=MONO_SMALL_FONT)
        y += max(26, bbox[3] - bbox[1] + 8)

    image.save(destination)
    return image


def metric_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str, value: str, accent: str = "#e7effc") -> None:
    rounded(draw, (x, y, x + w, y + h), fill=accent, radius=20)
    draw.text((x + 24, y + 20), label, font=SMALL_FONT, fill="#516173")
    draw.text((x + 24, y + 60), value, font=H2_FONT, fill="#18212b")


def browser_chrome(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    rounded(draw, (60, 90, 1540, 920), fill="#ffffff", outline="#d8e3f2", radius=28, width=2)
    draw.rectangle((60, 90, 1540, 150), fill="#f3f7fc")
    draw.ellipse((88, 112, 102, 126), fill="#ff5f57")
    draw.ellipse((112, 112, 126, 126), fill="#febc2e")
    draw.ellipse((136, 112, 150, 126), fill="#28c840")
    draw.text((190, 103), title, font=H2_FONT, fill="#18212b")
    draw.text((190, 132), subtitle, font=SMALL_FONT, fill="#64778c")


def save_report_preview(trace: RAGTrace, destination: Path) -> Image.Image:
    analysis = trace.analyze()
    image, draw = create_canvas()
    browser_chrome(draw, "RAG Debugger Report", "Shareable HTML output generated from a real trace")

    query_text = trace.query_input.text if trace.query_input else "No query recorded"
    draw.text((110, 182), query_text, font=TITLE_FONT, fill="#18212b")
    draw.text((110, 240), f"Trace: {trace.trace_id}  |  Pipeline: {trace.pipeline_name or 'n/a'}  |  Status: {trace.status}", font=SMALL_FONT, fill="#5a6d82")

    x_positions = [110, 390, 670, 950, 1230]
    metrics = [
        ("Overall health", f"{analysis.overall_health:.1f}"),
        ("Retrieval", f"{analysis.retrieval.score:.2f}"),
        ("Context", f"{analysis.context.score:.2f}"),
        ("Grounding", f"{analysis.grounding.score:.2f}"),
        ("Performance", f"{analysis.performance.score:.2f}"),
    ]
    for x, (label, value) in zip(x_positions, metrics):
        metric_card(draw, x, 290, 230, 120, label, value)

    rounded(draw, (110, 445, 760, 880), fill="#f7fafe", radius=24)
    draw.text((140, 475), "Recommendations", font=H2_FONT, fill="#18212b")
    bullet_text = "\n".join(f"- {item}" for item in analysis.recommendations or ["No recommendations"])
    draw_text_block(draw, bullet_text, (140, 525), BODY_FONT, "#243242", max_width=570, line_gap=10)

    rounded(draw, (800, 445, 1490, 880), fill="#f7fafe", radius=24)
    draw.text((830, 475), "Retrieved Chunks", font=H2_FONT, fill="#18212b")
    headers = ["Chunk", "Rank", "Similarity", "Source"]
    header_x = [830, 1040, 1145, 1285]
    for hx, header in zip(header_x, headers):
        draw.text((hx, 525), header, font=SMALL_FONT, fill="#5a6d82")

    y = 565
    for chunk in trace.retrieval_step.retrieved_chunks[:4] if trace.retrieval_step else []:
        draw.line((830, y - 10, 1455, y - 10), fill="#dfe8f4", width=2)
        draw.text((830, y), chunk.chunk_id, font=BODY_FONT, fill="#18212b")
        draw.text((1045, y), str(chunk.rank), font=BODY_FONT, fill="#18212b")
        draw.text((1145, y), f"{chunk.similarity_score:.2f}", font=BODY_FONT, fill="#18212b")
        draw.text((1285, y), chunk.source, font=BODY_FONT, fill="#18212b")
        y += 58

    if trace.reranker_step:
        draw.text((830, 775), f"Reranker: {trace.reranker_step.reranker_name}", font=SMALL_FONT, fill="#5a6d82")
        if trace.reranker_step.reranked_chunks:
            top = trace.reranker_step.reranked_chunks[0]
            draw.text(
                (830, 812),
                f"Top reranked chunk {top.chunk_id} moved {top.original_rank}->{top.reranked_rank} with score {top.reranker_score:.2f}",
                font=SMALL_FONT,
                fill="#243242",
            )

    image.save(destination)
    return image


def save_dashboard_preview(traces: Sequence[RAGTrace], destination: Path) -> Image.Image:
    image, draw = create_canvas(background="#f3f8ff")
    browser_chrome(draw, "RAG Debugger Dashboard", "Static dashboard built from the bundled example traces")

    analyses = [trace.analyze() for trace in traces]
    avg_health = sum(item.overall_health for item in analyses) / len(analyses)
    avg_retrieval = sum(item.retrieval.score for item in analyses) / len(analyses)
    avg_grounding = sum(item.grounding.score for item in analyses) / len(analyses)
    avg_latency = sum((trace.summary.total_latency_ms if trace.summary else 0) for trace in traces) / len(traces)

    cards = [
        ("Trace count", str(len(traces))),
        ("Avg health", f"{avg_health:.1f}"),
        ("Avg retrieval", f"{avg_retrieval:.2f}"),
        ("Avg grounding", f"{avg_grounding:.2f}"),
        ("Avg latency ms", f"{avg_latency:.0f}"),
    ]
    x = 110
    for label, value in cards:
        metric_card(draw, x, 210, 250, 112, label, value, accent="#eef4fd")
        x += 275

    rounded(draw, (110, 360, 1490, 880), fill="#ffffff", radius=24)
    draw.text((140, 390), "Pipeline snapshot", font=H2_FONT, fill="#18212b")
    columns = [140, 420, 670, 900, 1115, 1280]
    headers = ["Trace", "Pipeline", "Health", "Retrieval", "Grounding", "Latency"]
    for cx, header in zip(columns, headers):
        draw.text((cx, 438), header, font=SMALL_FONT, fill="#5a6d82")

    y = 488
    for trace, analysis in zip(traces, analyses):
        draw.line((140, y - 14, 1450, y - 14), fill="#e2ebf7", width=2)
        draw.text((140, y), trace.trace_id[:16], font=BODY_FONT, fill="#18212b")
        draw.text((420, y), (trace.pipeline_name or "n/a")[:18], font=BODY_FONT, fill="#18212b")
        draw.text((670, y), f"{analysis.overall_health:.1f}", font=BODY_FONT, fill="#18212b")
        draw.text((900, y), f"{analysis.retrieval.score:.2f}", font=BODY_FONT, fill="#18212b")
        draw.text((1115, y), f"{analysis.grounding.score:.2f}", font=BODY_FONT, fill="#18212b")
        draw.text((1280, y), f"{trace.summary.total_latency_ms if trace.summary else 0}", font=BODY_FONT, fill="#18212b")
        y += 82

    report = build_team_report(traces, group_by="tag")
    if report.top_findings:
        draw.text((140, 770), "Top findings", font=SMALL_FONT, fill="#5a6d82")
        bullet_text = "\n".join(f"- {item.finding} ({item.count})" for item in report.top_findings[:3])
        draw_text_block(draw, bullet_text, (140, 804), SMALL_FONT, "#243242", max_width=1260, line_gap=8)

    image.save(destination)
    return image


def add_frame_label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    rounded(draw, (1180, 34, 1530, 88), fill="#14253d", radius=18)
    draw.text((1206, 47), text, font=SMALL_FONT, fill="#f7fbff")
    return labeled


def save_walkthrough_gif(frames: Iterable[Image.Image], destination: Path) -> None:
    rendered_frames = []
    labels = ["CLI analyze", "HTML report", "Dashboard"]
    for label, frame in zip(labels, frames):
        rendered_frames.append(add_frame_label(frame, label))

    rendered_frames[0].save(
        destination,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=[1500, 1500, 1500],
        loop=0,
    )


def main() -> None:
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    EXAMPLE_ASSETS.mkdir(parents=True, exist_ok=True)

    cli_output = run_command(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "rag-debug"),
            "analyze",
            "examples/healthy_trace.json",
        ]
    )
    run_command(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "rag-debug"),
            "export",
            "examples/healthy_trace.json",
            "--format",
            "html",
            "--output",
            str(EXAMPLE_ASSETS / "report.html"),
        ]
    )
    run_command(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "rag-debug"),
            "dashboard",
            "examples/",
            "--output",
            str(EXAMPLE_ASSETS / "dashboard.html"),
        ]
    )

    traces = [
        RAGTrace.load(PROJECT_ROOT / "examples" / "healthy_trace.json"),
        RAGTrace.load(PROJECT_ROOT / "examples" / "low_retrieval_trace.json"),
        RAGTrace.load(PROJECT_ROOT / "examples" / "hallucination_trace.json"),
    ]

    cli_image = save_terminal_capture(
        "rag-debug analyze examples/healthy_trace.json",
        cli_output,
        DOCS_ASSETS / "cli-analyze.png",
    )
    report_image = save_report_preview(traces[0], DOCS_ASSETS / "report-preview.png")
    dashboard_image = save_dashboard_preview(traces, DOCS_ASSETS / "dashboard-preview.png")
    save_walkthrough_gif(
        [cli_image, report_image, dashboard_image],
        DOCS_ASSETS / "walkthrough.gif",
    )


if __name__ == "__main__":
    main()
