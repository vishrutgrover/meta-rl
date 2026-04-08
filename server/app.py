"""FastAPI application for the GTM Strategy Optimizer environment."""

from __future__ import annotations

import os
import sys

# Ensure parent directory is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from models import GTMAction, GTMObservation
from server.environment import GTMEnvironment

app = create_app(GTMEnvironment, GTMAction, GTMObservation)

# ── Server entry point ─────────────────────────────────────────────────────

def main() -> None:
    """Run the FastAPI server with uvicorn (used as a console script)."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
