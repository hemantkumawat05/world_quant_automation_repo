"""FastAPI application.

Run with::

    uv run uvicorn alpha_harness.main:app --reload --port 8000

The frontend is a separate process and talks to this over HTTP + WebSocket only.
Credentials never leave the backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import (
    alphas,
    auth,
    catalog,
    chat,
    diversify,
    ga,
    harden,
    harvest,
    invent,
    lab_tasks,
    labs,
    llm,
    optimize,
    pair,
    plan,
    power_pool_lab,
    relocate,
    repair,
    search_lab,
    sims,
    tasks,
    template_lab,
    templates,
    today,
    vault,
    ws,
)
from .api.deps import install_exception_handlers
from .config import Settings, get_settings
from .logging import configure_logging
from .state import AppState

log = structlog.get_logger(__name__)

DESCRIPTION = """
Local research studio for the WorldQuant BRAIN platform.

Runs entirely on your machine. BRAIN credentials are encrypted at rest and never
reach the browser.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(settings)
        app.state.harness = state
        await state.startup()
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(
        title="Alpha Harness",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins if "*" not in settings.cors_origins else None,
        allow_origin_regex=r"https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)

    app.include_router(auth.router)
    app.include_router(catalog.router)
    app.include_router(sims.router)
    app.include_router(alphas.router)
    app.include_router(templates.router)
    app.include_router(template_lab.router)
    app.include_router(ga.router)
    app.include_router(optimize.router)
    app.include_router(llm.router)
    app.include_router(vault.router)
    app.include_router(harvest.router)
    app.include_router(plan.router)
    app.include_router(pair.router)
    app.include_router(invent.router)
    app.include_router(relocate.router)
    app.include_router(repair.router)
    app.include_router(harden.router)
    app.include_router(diversify.router)
    app.include_router(tasks.router)
    app.include_router(today.router)
    app.include_router(labs.router)
    app.include_router(search_lab.router)
    app.include_router(lab_tasks.router)
    app.include_router(power_pool_lab.router)
    app.include_router(chat.router)
    app.include_router(ws.router)

    @app.get("/", tags=["meta"])
    @app.get("/health", tags=["meta"])
    @app.get("/api/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        """Liveness plus a summary of local state, for the UI's status bar."""
        state: AppState = app.state.harness
        return {
            "ok": True,
            "version": app.version,
            "dataDir": str(state.settings.data_dir),
            "database": await state.db.healthcheck(),
            "session": state.auth.session.to_dict(),
            "activeSimulations": len(await state.tracker.active()),
            "websocketClients": state.hub.client_count,
        }

    return app


app = create_app()
