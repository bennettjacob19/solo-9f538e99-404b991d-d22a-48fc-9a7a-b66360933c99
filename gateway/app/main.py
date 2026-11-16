"""FastAPI entrypoint for the embedding batch gateway."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .batcher import BatchScheduler
from .config import Settings
from .corpus import CorpusIndex
from .errors import GatewayError, ValidationError
from .schemas import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    IndexQueryRequest,
    IndexQueryResponse,
    IndexStatusResponse,
    QueryMatch,
    StatsResponse,
)

logger = logging.getLogger("embedding_gateway")


class AcceptanceTimeMiddleware:
    """Stamp the receipt time before the body is parsed.

    The per-request deadline must start when the gateway accepts the call,
    not when the request reaches the scheduler.  Index queries additionally
    capture the current index snapshot here, before the body is parsed, so a
    build that finishes while the request body is still being read cannot
    change which index the accepted query searches.
    """

    #: Path whose index snapshot is captured at acceptance.
    SNAPSHOT_PATH = "/v1/index/query"

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            state = scope.setdefault("state", {})
            state.setdefault("accepted_at", asyncio.get_running_loop().time())
            if scope.get("path") == self.SNAPSHOT_PATH:
                app = scope.get("app")
                index = getattr(getattr(app, "state", None), "index", None)
                # Always record the capture, even when it is ``None``: that
                # pins "no index existed at acceptance" instead of silently
                # letting a build that finishes during body parsing change
                # which index the query sees.
                state["index_snapshot_at_acceptance"] = (
                    index.snapshot if index is not None else None
                )
        await self.app(scope, receive, send)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


def create_app(
    settings: Settings | None = None,
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            settings.backend_timeout_seconds,
            connect=settings.backend_connect_timeout_seconds,
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            transport=backend_transport,
            limits=httpx.Limits(
                max_connections=1, max_keepalive_connections=1
            ),
        )
        scheduler = BatchScheduler(settings, client)
        index = CorpusIndex(settings, scheduler, client)
        app.state.client = client
        app.state.scheduler = scheduler
        app.state.index = index
        await scheduler.start()
        logger.info(
            "gateway ready: backend=%s window_ms=%s batch_max=%s",
            settings.backend_url,
            settings.batch_window_ms,
            settings.max_texts_per_batch,
        )
        try:
            yield
        finally:
            await scheduler.close()
            await client.aclose()

    app = FastAPI(
        title="Embedding Batch Gateway",
        version="1.0.0",
        description=(
            "Accepts 1-4 short texts per call, merges concurrent calls into "
            "single-flight backend batches of at most 8 texts, and enforces a "
            "deadline measured from request acceptance. Also maintains an "
            "in-memory corpus index (max 32 documents) whose build and query "
            "operations pin the backend model revision, so vectors from "
            "different model revisions are never mixed."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(AcceptanceTimeMiddleware)
    app.state.settings = settings

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        request: Request, exc: GatewayError
    ) -> JSONResponse:
        return _error_response(
            exc.status_code, exc.code, exc.message, exc.details
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        first = errors[0] if errors else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{location}: {first.get('msg', 'invalid request')}"
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": message.lstrip(": "),
                    "details": errors,
                }
            },
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def stats(request: Request) -> StatsResponse:
        return StatsResponse(
            **request.app.state.scheduler.stats(),
            **request.app.state.index.stats(),
        )

    def _resolve_deadline(payload: Any, request: Request) -> float:
        timeout_ms = (
            payload.timeout_ms
            if payload.timeout_ms is not None
            else settings.default_timeout_ms
        )
        if timeout_ms > settings.max_timeout_ms:
            raise ValidationError(
                f"timeout_ms must be <= {settings.max_timeout_ms}"
            )
        return request.state.accepted_at + timeout_ms / 1000.0

    @app.post(
        "/v1/index/build",
        response_model=IndexBuildResponse,
        tags=["index"],
        summary="Atomically replace the in-memory corpus index",
    )
    async def index_build(
        payload: IndexBuildRequest, request: Request
    ) -> IndexBuildResponse:
        seen: set[str] = set()
        documents: list[tuple[str, str]] = []
        for document in payload.documents:
            doc_id = document.id.strip()
            if not doc_id:
                raise ValidationError("document ids must not be blank")
            if doc_id in seen:
                raise ValidationError(
                    f"duplicate document id {doc_id!r}; ids must be unique "
                    "within one corpus build"
                )
            seen.add(doc_id)
            documents.append((doc_id, document.text))

        deadline = _resolve_deadline(payload, request)
        model = payload.model or settings.backend_model
        outcome = await request.app.state.index.build(
            documents, model=model, deadline=deadline
        )
        snapshot = outcome.snapshot
        return IndexBuildResponse(
            index_version=snapshot.version,
            model_revision=snapshot.revision,
            model=snapshot.model,
            dimension=snapshot.dim,
            document_count=len(snapshot.documents),
            batch_ids=list(outcome.batch_ids),
        )

    @app.post(
        "/v1/index/query",
        response_model=IndexQueryResponse,
        tags=["index"],
        summary="Return the 3 closest documents in the pinned index",
    )
    async def index_query(
        payload: IndexQueryRequest, request: Request
    ) -> IndexQueryResponse:
        deadline = _resolve_deadline(payload, request)
        outcome = await request.app.state.index.query(
            payload.query,
            model=payload.model,
            deadline=deadline,
            request=request,
            snapshot=getattr(
                request.state, "index_snapshot_at_acceptance", None
            ),
            snapshot_pinned=hasattr(
                request.state, "index_snapshot_at_acceptance"
            ),
        )
        return IndexQueryResponse(
            index_version=outcome.snapshot.version,
            model_revision=outcome.snapshot.revision,
            model=outcome.snapshot.model,
            matches=[
                QueryMatch(id=match.doc_id, text=match.text, score=match.score)
                for match in outcome.matches
            ],
            batch_id=outcome.batch_id,
            batch_size=outcome.batch_size,
        )

    @app.get(
        "/v1/index/status",
        response_model=IndexStatusResponse,
        tags=["index"],
        summary="Describe the installed index (if any)",
    )
    async def index_status(request: Request) -> IndexStatusResponse:
        index = request.app.state.index
        stats = index.stats()
        snapshot = index.snapshot
        return IndexStatusResponse(
            indexed=snapshot is not None,
            index_version=snapshot.version if snapshot else None,
            document_count=len(snapshot.documents) if snapshot else None,
            model_revision=snapshot.revision if snapshot else None,
            model=snapshot.model if snapshot else None,
            dimension=snapshot.dim if snapshot else None,
            created_ago_seconds=(
                round(time.monotonic() - snapshot.created_at, 3)
                if snapshot
                else None
            ),
            index_builds_completed=stats["index_builds_completed"],
            index_builds_failed=stats["index_builds_failed"],
            index_queries_completed=stats["index_queries_completed"],
            index_queries_rejected=stats["index_queries_rejected"],
        )

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        tags=["embeddings"],
        summary="Embed 1-4 texts",
    )
    async def embeddings(
        payload: EmbeddingRequest, request: Request
    ) -> EmbeddingResponse:
        texts = payload.as_texts()
        if not 1 <= len(texts) <= settings.max_texts_per_request:
            raise ValidationError(
                f"input must contain between 1 and "
                f"{settings.max_texts_per_request} texts"
            )

        timeout_ms = (
            payload.timeout_ms
            if payload.timeout_ms is not None
            else settings.default_timeout_ms
        )
        if timeout_ms > settings.max_timeout_ms:
            raise ValidationError(
                f"timeout_ms must be <= {settings.max_timeout_ms}"
            )

        accepted_at = request.state.accepted_at
        deadline = accepted_at + timeout_ms / 1000.0
        model = payload.model or settings.backend_model
        request_id = uuid.uuid4().hex

        result = await request.app.state.scheduler.submit(
            request_id=request_id,
            texts=texts,
            model=model,
            deadline=deadline,
            request=request,
        )
        return EmbeddingResponse(
            data=[
                EmbeddingItem(index=index, embedding=vector)
                for index, vector in enumerate(result.vectors)
            ],
            model=model,
            batch_id=result.batch_id,
            batch_size=result.batch_size,
        )

    return app


app = create_app()
