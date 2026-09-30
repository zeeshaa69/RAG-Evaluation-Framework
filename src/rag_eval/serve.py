"""Tiny local HTTP server: static dashboard files plus a live JSON API."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .store import ResultStore


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, store: ResultStore, **kwargs: Any) -> None:
        self._store = store
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/bundle.json":
            body = json.dumps(self._store.bundle()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None


def make_server(
    results_dir: str | Path, dashboard_dir: str | Path, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    dash = Path(dashboard_dir)
    if not (dash / "index.html").is_file():
        raise FileNotFoundError(
            f"{dash}/index.html not found. Build the dashboard first: cd dashboard && npm run build"
        )
    handler = partial(_Handler, directory=str(dash), store=ResultStore(results_dir))
    return ThreadingHTTPServer((host, port), handler)


def export_static(results_dir: str | Path, dashboard_dir: str | Path, out_dir: str | Path) -> Path:
    """Copy the built dashboard to ``out_dir`` with a frozen ``api/bundle.json``."""
    import shutil

    dash, out = Path(dashboard_dir), Path(out_dir)
    if not (dash / "index.html").is_file():
        raise FileNotFoundError(f"{dash}/index.html not found; build the dashboard first")
    shutil.copytree(dash, out, dirs_exist_ok=True)
    (out / "api").mkdir(exist_ok=True)
    (out / "api" / "bundle.json").write_text(
        json.dumps(ResultStore(results_dir).bundle()), encoding="utf-8"
    )
    return out
