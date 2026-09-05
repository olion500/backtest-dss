from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web.api.app.schemas import BacktestRequest
from web.api.app.services.backtest import default_request, run_backtest_view, run_order_book_view
from web.api.app.services.market_data import MarketDataError


app = FastAPI(
    title="Dongpa Viewer API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

origins = [origin.strip() for origin in os.getenv("DONGPA_CORS_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

rate_limit = max(1, int(os.getenv("DONGPA_RATE_LIMIT", "12")))
rate_window_seconds = 60
request_times: dict[str, deque[float]] = defaultdict(deque)
rate_lock = threading.Lock()


@app.middleware("http")
async def protect_public_compute(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/v1/"):
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with rate_lock:
            history = request_times[client]
            while history and now - history[0] >= rate_window_seconds:
                history.popleft()
            if len(history) >= rate_limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "계산 요청이 너무 많습니다. 잠시 후 다시 시도하세요."},
                )
            history.append(now)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/config/defaults")
def defaults() -> dict:
    return default_request().model_dump(mode="json")


def _run_view(handler, request: BacktestRequest) -> dict:
    try:
        return handler(request)
    except MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/backtests/run")
def backtest(request: BacktestRequest) -> dict:
    return _run_view(run_backtest_view, request)


@app.post("/api/v1/order-book/preview")
def order_book(request: BacktestRequest) -> dict:
    return _run_view(run_order_book_view, request)


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets_path = FRONTEND_DIST / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
