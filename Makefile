.PHONY: setup dev api web test lint eval build docker

setup:            ## install backend + frontend dependencies
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
	cd frontend && npm ci

api:              ## run the API with auto-reload on :8000
	. .venv/bin/activate && uvicorn finpilot.main:app --reload --port 8000

web:              ## run the Vite dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test:             ## backend test suite
	. .venv/bin/activate && pytest

lint:             ## lint + format check + frontend typecheck
	. .venv/bin/activate && ruff check . && ruff format --check .
	cd frontend && npx tsc -b

eval:             ## regenerate evaluation/results (deterministic, no API cost)
	. .venv/bin/activate && python -m evaluation.run_eval

build:            ## production frontend build (served by FastAPI from frontend/dist)
	cd frontend && npm run build

docker:           ## build and run the container
	docker compose up --build
