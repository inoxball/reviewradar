"""``reviewradar serve``: the demo web panel."""

from typing import Annotated

import typer

from reviewradar.cli._common import fail
from reviewradar.config import get_settings


def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = 8000,
) -> None:
    """Start the web panel: search, mode comparison, corpus overview and evaluation."""
    try:
        import uvicorn

        from reviewradar.web.app import create_app
        from reviewradar.web.wiring import build_panel_services
    except ImportError:
        fail("the web panel needs the optional 'web' extra: pip install -e '.[web]'")

    settings = get_settings()
    app = create_app(lambda: build_panel_services(settings))
    typer.echo(f"ReviewRadar panel: http://{host}:{port}  (API docs: http://{host}:{port}/docs)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
