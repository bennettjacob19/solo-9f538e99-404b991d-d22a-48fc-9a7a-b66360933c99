"""In-memory corpus index with revision-pinned builds and queries.

Semantics
---------
* A build accepts the **whole** corpus (at most 32 documents with unique
  ids) and pins the backend model revision at acceptance time.  All document
  embeddings are obtained through the normal single-flight
  :class:`~gateway.app.batcher.BatchScheduler`; the index is installed with a
  single atomic assignment only after every vector is present and every
  backend-reported revision equals the pinned one.  Any failure leaves the
  previously installed snapshot untouched.
* A query captures the current snapshot (content and revision) when the HTTP
  request is accepted, before its body is parsed; direct callers capture it
  before the first ``await``.  A build that replaces the snapshot while the
  query is waiting for inference therefore has no effect on that query.  The
  query embedding's actual revision must equal the snapshot revision;
  otherwise the query fails explicitly instead of falling back to old vectors
  or a newer index.
* Snapshots are immutable tuples, so a query can safely reuse a captured
  snapshot after a swap.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .batcher import BatchResult, BatchScheduler
from .config import Settings
from .errors import (
    BackendError,
    GatewayError,
    IndexIncompatible,
    IndexUnavailable,
    RequestTimeout,
    RevisionMismatch,
    RevisionUnavailable,
)

logger = logging.getLogger("embedding_gateway.corpus")


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class IndexSnapshot:
    version: int
    revision: str
    model: str
    dim: int
    documents: tuple[IndexedDocument, ...]
    created_at: float


@dataclass(frozen=True)
class QueryMatch:
    doc_id: str
    text: str
    score: float


@dataclass(frozen=True)
class BuildOutcome:
    snapshot: IndexSnapshot
    batch_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryOutcome:
    snapshot: IndexSnapshot
    query_revision: str
    matches: tuple[QueryMatch, ...]
    batch_id: str
    batch_size: int


class CorpusIndex:
    """Holds at most one immutable corpus snapshot at a time."""

    def __init__(
        self,
        settings: Settings,
        scheduler: BatchScheduler,
        client: httpx.AsyncClient,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler
        # The scheduler deliberately uses a single-connection client.  Sharing
        # it for the tiny revision lookup means the lookup can never run
        # concurrently with an inference batch, so the backend still sees at
        # most one HTTP request at a time.
        self.client = client
        self._snapshot: IndexSnapshot | None = None
        self._version = 0
        self._build_lock = asyncio.Lock()
        self._stats: dict[str, int] = {
            "index_builds_completed": 0,
            "index_builds_failed": 0,
            "index_queries_completed": 0,
            "index_queries_rejected": 0,
        }

    # ---------------------------------------------------------------- state
    @property
    def snapshot(self) -> IndexSnapshot | None:
        return self._snapshot

    def stats(self) -> dict[str, Any]:
        snapshot = self._snapshot
        result: dict[str, Any] = dict(self._stats)
        result.update(
            {
                "index_version": snapshot.version if snapshot else None,
                "index_documents": (
                    len(snapshot.documents) if snapshot else None
                ),
                "index_revision": snapshot.revision if snapshot else None,
            }
        )
        return result

    def _install(
        self,
        *,
        revision: str,
        model: str,
        dim: int,
        documents: tuple[IndexedDocument, ...],
    ) -> IndexSnapshot:
        """Atomically replace the current snapshot (no ``await`` inside)."""
        self._version += 1
        snapshot = IndexSnapshot(
            version=self._version,
            revision=revision,
            model=model,
            dim=dim,
            documents=documents,
            created_at=time.monotonic(),
        )
        self._snapshot = snapshot
        logger.info(
            "installed corpus index version=%s documents=%s revision=%s dim=%s",
            snapshot.version,
            len(documents),
            revision,
            dim,
        )
        return snapshot

    # ------------------------------------------------------------- revisions
    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.settings.backend_api_key:
            headers["authorization"] = f"Bearer {self.settings.backend_api_key}"
        return headers

    async def current_revision(self, deadline: float) -> str:
        """Read the backend's current revision and pin it.

        Called once when a build is accepted, before any document embedding is
        scheduled.  The request is metadata-only and does not consume the
        backend's inference slot; sharing the scheduler's single-connection
        client additionally guarantees it never overlaps an inference batch.
        """
        loop = asyncio.get_running_loop()
        url = self.settings.backend_revision_url
        if not url:
            raise RevisionUnavailable(
                "the backend revision endpoint is not configured "
                "(EMBEDDING_BACKEND_REVISION_URL), so an index cannot be "
                "pinned to a model revision"
            )

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RequestTimeout(
                "request deadline elapsed before the model revision was read"
            )
        timeout = max(
            0.001, min(remaining, self.settings.backend_timeout_seconds)
        )
        try:
            response = await self.client.get(
                url, headers=self._headers(), timeout=timeout
            )
        except httpx.TimeoutException as exc:
            if loop.time() >= deadline:
                raise RequestTimeout(
                    "request deadline elapsed while reading the model revision"
                ) from exc
            raise BackendError(
                f"backend revision request timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(
                f"backend revision request failed: {exc}"
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text[:500].replace("\n", " ")
            raise RevisionUnavailable(
                "backend does not provide a usable model revision "
                f"(HTTP {response.status_code}: {detail})"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RevisionUnavailable(
                "backend revision response is not valid JSON"
            ) from exc
        revision = body.get("revision") if isinstance(body, dict) else None
        if not isinstance(revision, str) or not revision.strip():
            raise RevisionUnavailable(
                "backend revision response is missing a non-empty 'revision'"
            )
        return revision

    # ----------------------------------------------------------------- build
    async def build(
        self,
        documents: list[tuple[str, str]],
        *,
        model: str,
        deadline: float,
    ) -> BuildOutcome:
        """Embed and atomically install the whole corpus.

        ``documents`` is a list of ``(id, text)`` pairs that has already been
        validated (1..32, unique ids, non-empty).
        """
        try:
            outcome = await self._build(
                documents, model=model, deadline=deadline
            )
        except BaseException:
            self._stats["index_builds_failed"] += 1
            raise
        self._stats["index_builds_completed"] += 1
        return outcome

    async def _build(
        self,
        documents: list[tuple[str, str]],
        *,
        model: str,
        deadline: float,
    ) -> BuildOutcome:
        loop = asyncio.get_running_loop()
        # Pin the revision at acceptance, before waiting for the build lock or
        # any embedding inference.
        pinned_revision = await self.current_revision(deadline)

        await self._acquire_build_lock(deadline)
        try:
            if loop.time() >= deadline:
                raise RequestTimeout(
                    "request deadline elapsed before the corpus build started"
                )
            chunk_size = max(1, self.settings.max_texts_per_batch)
            chunks = [
                documents[i : i + chunk_size]
                for i in range(0, len(documents), chunk_size)
            ]
            tasks = [
                asyncio.create_task(
                    self._embed_build_chunk(chunk, model=model, deadline=deadline),
                    name=f"index-build-{position}",
                )
                for position, chunk in enumerate(chunks)
            ]
            try:
                raw = await asyncio.gather(*tasks, return_exceptions=True)
            except BaseException:
                # The HTTP handler was cancelled; make sure no chunk stays
                # queued in the scheduler.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            results: list[BatchResult] = []
            pending_error: GatewayError | None = None
            for outcome in raw:
                if isinstance(outcome, GatewayError):
                    if pending_error is None:
                        pending_error = outcome
                elif isinstance(outcome, BaseException):
                    if pending_error is None:
                        pending_error = BackendError(
                            f"corpus build failed: {outcome}"
                        )
                else:
                    results.append(outcome)
            if pending_error is not None:
                raise pending_error
            if len(results) != len(chunks):
                raise BackendError(
                    "corpus build did not return a result for every chunk"
                )

            observed_revisions: set[str] = set()
            vectors: list[list[float]] = []
            batch_ids: list[str] = []
            for result, chunk in zip(results, chunks):
                if result.revision is None:
                    raise RevisionUnavailable(
                        "backend did not report a model revision for the "
                        "corpus build; the previous index was kept"
                    )
                observed_revisions.add(result.revision)
                if result.revision != pinned_revision:
                    raise RevisionMismatch(
                        "backend model revision changed during the corpus "
                        f"build (pinned {pinned_revision!r}, observed "
                        f"{sorted(observed_revisions)!r}); the build was "
                        "rejected and the previous index was kept",
                        details={
                            "pinned_revision": pinned_revision,
                            "observed_revisions": sorted(observed_revisions),
                        },
                    )
                if len(result.vectors) != len(chunk):
                    raise BackendError(
                        "backend returned an incomplete vector set for the "
                        "corpus build"
                    )
                vectors.extend(result.vectors)
                if result.batch_id not in batch_ids:
                    batch_ids.append(result.batch_id)

            if len(vectors) != len(documents):
                raise BackendError(
                    "corpus build is missing document embeddings"
                )
            dim = len(vectors[0]) if vectors else 0
            if dim < 1 or any(len(vector) != dim for vector in vectors):
                raise BackendError(
                    "backend returned missing or inconsistent embedding "
                    "dimensions for the corpus build"
                )
            if any(
                not math.isfinite(value)
                for vector in vectors
                for value in vector
            ):
                raise BackendError(
                    "backend returned non-finite values for the corpus build"
                )

            indexed = tuple(
                IndexedDocument(
                    doc_id=doc_id,
                    text=text,
                    vector=tuple(vector),
                )
                for (doc_id, text), vector in zip(documents, vectors)
            )
            snapshot = self._install(
                revision=pinned_revision,
                model=model,
                dim=dim,
                documents=indexed,
            )
            return BuildOutcome(snapshot=snapshot, batch_ids=tuple(batch_ids))
        finally:
            self._build_lock.release()

    async def _acquire_build_lock(self, deadline: float) -> None:
        """Acquire the build lock, bounded by the request deadline.

        ``asyncio.wait_for(lock.acquire(), ...)`` can lose a race in which the
        lock is granted exactly as the timeout fires, leaving the lock held
        with nobody to release it.  The acquisition task is therefore inspected
        after cancellation: if it did acquire, the lock is released here.
        """
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RequestTimeout(
                "request deadline elapsed before the corpus build could start"
            )

        acquire = asyncio.ensure_future(self._build_lock.acquire())
        try:
            done, _ = await asyncio.wait({acquire}, timeout=remaining)
        except BaseException:
            # The caller was cancelled: make sure a lock granted at the same
            # moment is not leaked.
            acquire.cancel()
            try:
                acquired = await acquire
            except asyncio.CancelledError:
                acquired = False
            if acquired:
                self._build_lock.release()
            raise

        if acquire in done and acquire.result():
            return

        # Timed out.  Cancel the waiter and, if it still managed to take the
        # lock, hand it back before failing the build.
        acquire.cancel()
        try:
            acquired = await acquire
        except asyncio.CancelledError:
            acquired = False
        if acquired:
            self._build_lock.release()

        raise RequestTimeout(
            "request deadline elapsed while another corpus build was running"
        )

    async def _embed_build_chunk(
        self,
        chunk: list[tuple[str, str]],
        *,
        model: str,
        deadline: float,
    ) -> BatchResult:
        return await self.scheduler.submit(
            request_id=uuid.uuid4().hex,
            texts=[text for _, text in chunk],
            model=model,
            deadline=deadline,
        )

    # ----------------------------------------------------------------- query
    async def query(
        self,
        query_text: str,
        *,
        model: str | None,
        deadline: float,
        request: Any | None = None,
        snapshot: IndexSnapshot | None = None,
        snapshot_pinned: bool = False,
    ) -> QueryOutcome:
        """Embed ``query_text`` and rank the pinned snapshot.

        ``snapshot`` is the index captured when the HTTP request was accepted
        (before body parsing); ``snapshot_pinned`` distinguishes "captured
        before any index existed" from "not captured".  When not pinned, the
        current snapshot is captured synchronously before the first
        ``await``, which is enough for direct callers.  Either way, a build
        finishing later cannot affect this query.
        """
        try:
            outcome = await self._query(
                query_text,
                model=model,
                deadline=deadline,
                request=request,
                snapshot=snapshot,
                snapshot_pinned=snapshot_pinned,
            )
        except BaseException:
            self._stats["index_queries_rejected"] += 1
            raise
        self._stats["index_queries_completed"] += 1
        return outcome

    async def _query(
        self,
        query_text: str,
        *,
        model: str | None,
        deadline: float,
        request: Any | None,
        snapshot: IndexSnapshot | None,
        snapshot_pinned: bool,
    ) -> QueryOutcome:
        # Pin the index snapshot (content + revision) before the first await.
        # A build completing later cannot affect this query.
        if not snapshot_pinned:
            snapshot = self._snapshot
        if snapshot is None:
            raise IndexUnavailable(
                "no corpus index has been built; query rejected"
            )
        if model is not None and model != snapshot.model:
            raise IndexIncompatible(
                f"query model {model!r} does not match index version "
                f"{snapshot.version} model {snapshot.model!r}"
            )

        result = await self.scheduler.submit(
            request_id=uuid.uuid4().hex,
            texts=[query_text],
            model=snapshot.model,
            deadline=deadline,
            request=request,
        )
        if result.revision is None:
            raise RevisionUnavailable(
                "backend did not report a model revision for the query "
                f"embedding; index version {snapshot.version} "
                f"({snapshot.revision!r}) is unavailable for this query"
            )
        if result.revision != snapshot.revision:
            raise RevisionMismatch(
                "query embedding was produced with model revision "
                f"{result.revision!r} but index version {snapshot.version} "
                f"is pinned to {snapshot.revision!r}; the index is "
                "unavailable for this query (no substitute index or old "
                "vector was used)",
                details={
                    "index_version": snapshot.version,
                    "index_revision": snapshot.revision,
                    "query_revision": result.revision,
                },
            )
        if len(result.vectors) != 1:
            raise BackendError(
                "backend returned an unexpected number of query embeddings"
            )
        query_vector = result.vectors[0]
        if len(query_vector) != snapshot.dim:
            raise IndexIncompatible(
                f"query embedding has dimension {len(query_vector)} but "
                f"index version {snapshot.version} has dimension "
                f"{snapshot.dim}"
            )

        ranked = self._rank(snapshot, query_vector)
        return QueryOutcome(
            snapshot=snapshot,
            query_revision=result.revision,
            matches=tuple(ranked),
            batch_id=result.batch_id,
            batch_size=result.batch_size,
        )

    @staticmethod
    def _rank(
        snapshot: IndexSnapshot, query_vector: list[float]
    ) -> list[QueryMatch]:
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scored: list[tuple[float, int, IndexedDocument]] = []
        for position, document in enumerate(snapshot.documents):
            if query_norm == 0.0:
                score = 0.0
            else:
                doc_norm = math.sqrt(
                    sum(value * value for value in document.vector)
                )
                if doc_norm == 0.0:
                    score = 0.0
                else:
                    dot = sum(
                        left * right
                        for left, right in zip(
                            query_vector, document.vector
                        )
                    )
                    score = dot / (query_norm * doc_norm)
            scored.append((score, position, document))
        # Highest score first; document order breaks ties deterministically.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            QueryMatch(
                doc_id=document.doc_id,
                text=document.text,
                score=round(score, 6),
            )
            for score, _, document in scored[:3]
        ]
