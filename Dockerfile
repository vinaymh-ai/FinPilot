# ── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python API serving the built SPA ───────────────────────────────
FROM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite:////data/finpilot.db \
    FRONTEND_DIST=/app/frontend/dist \
    LOG_JSON=true
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY finpilot/ ./finpilot/
COPY evaluation/ ./evaluation/
COPY data/ ./data/
COPY --from=web /web/dist ./frontend/dist
RUN useradd --create-home --uid 10001 finpilot && mkdir -p /data && chown -R finpilot /data /app
USER finpilot
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn finpilot.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
