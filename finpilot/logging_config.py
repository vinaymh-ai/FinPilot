"""Structured logging: console for humans, a rotating JSON-lines file for debugging and the UI.

Every record carries `request_id`, `user_id` and `run_id` (from context variables), plus any
structured `extra={...}` fields (agent, model, tokens, latency, status…). The file handler always
writes JSON lines so `/api/observability/logs` can filter and search them; set LOG_LEVEL=DEBUG to
also log raw model payloads.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STD_ATTRS = set(logging.LogRecord("x", 0, "x", 0, "x", None, None).__dict__) | {"message", "asctime"}
_CONTEXT = ("request_id", "user_id", "run_id")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        from finpilot.llm.client import current_run_id
        from finpilot.observability.context import current_request_id, current_user_id

        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        if not hasattr(record, "user_id"):
            record.user_id = current_user_id()
        if not hasattr(record, "run_id"):
            record.run_id = current_run_id()
        return True


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v
        for k, v in record.__dict__.items()
        if k not in _STD_ATTRS and k not in _CONTEXT and not k.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _CONTEXT:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s%(ctx)s: %(message)s%(fields)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        ctx = [f"{k[0]}={getattr(record, k)}" for k in _CONTEXT if getattr(record, k, None) is not None]
        record.ctx = f" [{' '.join(ctx)}]" if ctx else ""
        extras = {
            k: v
            for k, v in _extras(record).items()
            if k not in ("ctx", "fields") and not isinstance(v, (dict, list))
        }
        record.fields = (" | " + " ".join(f"{k}={v}" for k, v in extras.items()))[:400] if extras else ""
        return super().format(record)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: str | None = None,
    max_mb: int = 20,
    backups: int = 5,
) -> Path | None:
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    handlers: list[logging.Handler] = [console]
    path: Path | None = None
    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                path, maxBytes=max_mb * 1024 * 1024, backupCount=backups, encoding="utf-8", delay=True
            )
            fh.setFormatter(JsonFormatter())
            handlers.append(fh)
        except OSError as exc:  # read-only filesystem etc. — keep console logging
            print(f"FinPilot: cannot write log file {path}: {exc}", file=sys.stderr)
            path = None
    ctx = ContextFilter()
    for h in handlers:
        h.addFilter(ctx)
    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level.upper())
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "apscheduler.executors.default",
        "pdfminer",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # our middleware logs every request with timing, user and request id
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return path
