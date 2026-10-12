"""FastAPI application entry point: `uvicorn finpilot.main:app`."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from finpilot import __version__
from finpilot.api.deps import require_token
from finpilot.api.routes import agents, core, planning, settings, transactions
from finpilot.config import get_settings
from finpilot.db import init_db, session_scope
from finpilot.llm.client import get_router
from finpilot.logging_config import configure_logging
from finpilot.models import Transaction
from finpilot.services import repository as repo
from finpilot.services.scheduler import scheduler

log = logging.getLogger("finpilot")


def bootstrap() -> None:
    """Create tables, restore runtime LLM overrides, seed + analyse the demo account on first boot."""
    s = get_settings()
    init_db()
    with session_scope() as db:
        get_router().set_overrides(repo.get_setting(db, "llm_overrides", {}) or {})
        repo.get_or_create_user(db)
        empty = db.query(Transaction).first() is None
        if empty and s.seed_demo_data:
            from finpilot.services.seed import seed_demo

            seed_demo(db)
    if empty and s.seed_demo_data:
        from finpilot.agents.coordinator import Coordinator

        try:
            Coordinator().run_pipeline(trigger="manual")
        except Exception:  # noqa: BLE001
            log.exception("initial demo analysis failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    bootstrap()
    if s.scheduler_enabled:
        with session_scope() as db:
            cron = repo.get_setting(db, "digest_cron")
        scheduler.start(digest_cron=cron)
    log.info("FinPilot %s ready (LLM: %s)", __version__, get_router().describe("advisor"))
    yield
    scheduler.shutdown()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="FinPilot API",
        version=__version__,
        lifespan=lifespan,
        description="Proactive multi-agent personal-finance co-pilot. All LLM calls route through Mesh API "
        "by default (OpenAI-compatible gateway).",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    protected = [Depends(require_token)]
    app.include_router(core.public, prefix="/api", tags=["system"])
    for r, tag in (
        (core.router, "system"),
        (transactions.router, "transactions"),
        (planning.router, "planning"),
        (agents.router, "agents"),
        (settings.router, "settings"),
    ):
        app.include_router(r, prefix="/api", tags=[tag], dependencies=protected)

    dist = Path(s.frontend_dist)
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> FileResponse:
            candidate = dist / full_path
            if full_path and candidate.is_file() and dist in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
