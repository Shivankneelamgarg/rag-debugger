from __future__ import annotations

import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Callable

from rag_debugger.core.trace import RAGTrace


DashboardRenderer = Callable[[list[RAGTrace], int | None], str]
TraceLoader = Callable[[], list[RAGTrace]]


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_dashboard_server(
    traces_loader: TraceLoader,
    renderer: DashboardRenderer,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    refresh_seconds: int = 5,
) -> tuple[ThreadingHTTPServer, int]:
    resolved_port = port or _find_free_port()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            traces = traces_loader()
            body = renderer(traces, refresh_seconds)
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, resolved_port), Handler)
    return server, resolved_port


def serve_dashboard_forever(
    traces_loader: TraceLoader,
    renderer: DashboardRenderer,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    refresh_seconds: int = 5,
) -> int:
    server, resolved_port = start_dashboard_server(
        traces_loader,
        renderer,
        host=host,
        port=port,
        refresh_seconds=refresh_seconds,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return resolved_port


def serve_dashboard_in_background(
    traces_loader: TraceLoader,
    renderer: DashboardRenderer,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    refresh_seconds: int = 5,
) -> tuple[ThreadingHTTPServer, Thread, int]:
    server, resolved_port = start_dashboard_server(
        traces_loader,
        renderer,
        host=host,
        port=port,
        refresh_seconds=refresh_seconds,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, resolved_port
