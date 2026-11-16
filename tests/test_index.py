"""Corpus index build/query semantics: revisions, atomic swaps, pinning."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gateway.app.main import create_app
from tests.conftest import FakeBackend, gateway_client, make_settings, post, vector_for


def documents(count: int, prefix: str = "doc") -> list[dict[str, str]]:
    return [
        {"id": f"{prefix}-{i}", "text": f"{prefix} text {i}"} for i in range(count)
    ]


async def build(
    client: httpx.AsyncClient,
    docs: list[dict[str, str]],
    timeout_ms: int = 5000,
) -> httpx.Response:
    return await client.post(
        "/v1/index/build",
        json={"documents": docs, "timeout_ms": timeout_ms},
    )


async def query(
    client: httpx.AsyncClient, text: str, timeout_ms: int = 5000
) -> httpx.Response:
    return await client.post(
        "/v1/index/query", json={"query": text, "timeout_ms": timeout_ms}
    )


def expected_top(query_text: str, docs: list[dict[str, str]], revision: str):
    import math

    def score(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

    qv = vector_for(query_text, 4, revision)
    ranked = sorted(
        (
            (round(score(qv, vector_for(d["text"], 4, revision)), 6), i, d["id"])
            for i, d in enumerate(docs)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [item[2] for item in ranked[:3]]


@pytest.mark.asyncio
async def test_build_then_query_returns_top_three_with_version_and_revision() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        docs = documents(10)
        built = await build(client, docs)
        assert built.status_code == 200, built.text
        body = built.json()
        assert body["index_version"] == 1
        assert body["model_revision"] == "rev-1"
        assert body["document_count"] == 10
        assert body["dimension"] == 4
        # 10 documents at 4 texts per request are merged into at most 8 per
        # batch through the normal scheduler.
        assert all(size <= 8 for size in backend.batch_sizes)

        response = await query(client, "a query")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["index_version"] == 1
        assert payload["model_revision"] == "rev-1"
        assert len(payload["matches"]) == 3
        assert [m["id"] for m in payload["matches"]] == expected_top(
            "a query", docs, "rev-1"
        )
        scores = [m["score"] for m in payload["matches"]]
        assert scores == sorted(scores, reverse=True)

        status = (await client.get("/v1/index/status")).json()
        assert status["indexed"] is True
        assert status["index_version"] == 1
        assert status["document_count"] == 10
        assert status["model_revision"] == "rev-1"
        assert status["index_builds_completed"] == 1
        assert status["index_queries_completed"] == 1


@pytest.mark.asyncio
async def test_build_validation_limits_ids_and_size() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        too_many = await build(client, documents(33))
        assert too_many.status_code == 422
        empty = await build(client, [])
        assert empty.status_code == 422
        duplicate = await build(
            client,
            [
                {"id": "same", "text": "a"},
                {"id": "same", "text": "b"},
            ],
        )
        assert duplicate.status_code == 422
        assert "unique" in duplicate.json()["error"]["message"]
        blank = await build(client, [{"id": "  ", "text": "a"}])
        assert blank.status_code == 422
        blank_text = await build(client, [{"id": "x", "text": ""}])
        assert blank_text.status_code == 422
        assert backend.calls == []
        status = (await client.get("/v1/index/status")).json()
        assert status["indexed"] is False


@pytest.mark.asyncio
async def test_query_without_index_is_explicitly_rejected() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        response = await query(client, "anything")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "index_unavailable"
        assert backend.calls == []


@pytest.mark.asyncio
async def test_failed_rebuild_keeps_previous_index() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        first = documents(5, "old")
        assert (await build(client, first)).status_code == 200

        backend.status = 503
        failed = await build(client, documents(5, "new"))
        assert failed.status_code == 502
        backend.status = 200

        status = (await client.get("/v1/index/status")).json()
        assert status["index_version"] == 1
        assert status["index_builds_failed"] == 1
        answer = await query(client, "old text 1")
        assert answer.status_code == 200
        assert answer.json()["index_version"] == 1
        assert {m["text"] for m in answer.json()["matches"]} <= {
            d["text"] for d in first
        }

        rebuilt = await build(client, documents(3, "new"))
        assert rebuilt.status_code == 200
        assert rebuilt.json()["index_version"] == 2
        answer = await query(client, "new text 1")
        assert answer.json()["index_version"] == 2


@pytest.mark.asyncio
async def test_revision_change_during_build_rejects_whole_build() -> None:
    backend = FakeBackend()

    async def switch_after_first_batch(call_number: int) -> None:
        call = backend.calls[call_number - 1]
        if any(text.startswith("mixed") for text in call.texts):
            backend.revision = "rev-2"

    backend.on_call = switch_after_first_batch
    async with gateway_client(backend) as (_, client):
        assert (await build(client, documents(4, "old"))).status_code == 200

        # The revision is pinned at acceptance (rev-1); the backend switches to
        # rev-2 while the build's batches are running, so some vectors would
        # come from each revision.  The build must be rejected wholesale.
        response = await build(client, documents(12, "mixed"))
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "revision_mismatch"
        assert error["details"]["pinned_revision"] == "rev-1"
        assert "rev-2" in error["details"]["observed_revisions"]

        status = (await client.get("/v1/index/status")).json()
        assert status["index_version"] == 1
        assert status["model_revision"] == "rev-1"
        # While the backend is on rev-2 the kept rev-1 index is unavailable.
        assert (await query(client, "old text 2")).status_code == 409
        backend.revision = "rev-1"
        answer = await query(client, "old text 2")
        assert answer.status_code == 200
        assert answer.json()["index_version"] == 1


@pytest.mark.asyncio
async def test_query_with_mismatched_revision_is_unavailable_not_remapped() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        docs = documents(6)
        assert (await build(client, docs)).status_code == 200

        backend.revision = "rev-2"
        response = await query(client, "query after upgrade")
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "revision_mismatch"
        assert error["details"]["index_revision"] == "rev-1"
        assert error["details"]["query_revision"] == "rev-2"
        # The old index was not used for the new-revision query, and the
        # gateway did not silently build/pick a newer index either.
        status = (await client.get("/v1/index/status")).json()
        assert status["index_version"] == 1
        assert status["model_revision"] == "rev-1"
        assert status["index_queries_rejected"] == 1

        backend.revision = "rev-1"
        assert (await query(client, "back on rev-1")).status_code == 200


@pytest.mark.asyncio
async def test_query_rejected_when_backend_stops_reporting_revision() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        assert (await build(client, documents(3))).status_code == 200
        backend.report_revision = False
        response = await query(client, "q")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "revision_unavailable"


@pytest.mark.asyncio
async def test_build_rejected_when_backend_cannot_report_revision() -> None:
    backend = FakeBackend()
    async with gateway_client(
        backend, backend_revision_url=""
    ) as (_, client):
        response = await build(client, documents(3))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "revision_unavailable"
        assert backend.calls == []


@pytest.mark.asyncio
async def test_query_uses_snapshot_accepted_before_a_swap() -> None:
    backend = FakeBackend(delay=0.05)
    async with gateway_client(backend) as (_, client):
        old_docs = documents(32, "old")
        assert (await build(client, old_docs)).status_code == 200

        # A rebuild is accepted first and therefore installs version 2 before
        # the query's own inference runs (single-flight).  The query is
        # accepted while version 1 is still current, so it must answer from
        # version 1.
        rebuild = asyncio.create_task(build(client, documents(32, "new")))
        await asyncio.sleep(0.03)
        answer = await query(client, "old text 3")
        assert answer.status_code == 200, answer.text
        payload = answer.json()
        assert payload["index_version"] == 1
        assert payload["model_revision"] == "rev-1"
        assert {m["text"] for m in payload["matches"]} <= {
            d["text"] for d in old_docs
        }

        assert (await rebuild).status_code == 200
        latest = await query(client, "new text 3")
        assert latest.json()["index_version"] == 2
        assert {m["text"] for m in latest.json()["matches"]} <= {
            d["text"] for d in documents(32, "new")
        }


@pytest.mark.asyncio
async def test_build_and_query_share_single_flight_scheduler() -> None:
    backend = FakeBackend(delay=0.03)
    async with gateway_client(backend) as (_, client):
        docs = documents(20)
        responses = await asyncio.gather(
            build(client, docs),
            post(client, ["plain", "embedding"]),
        )
        assert all(r.status_code == 200 for r in responses)
        assert backend.max_active == 1
        assert all(size <= 8 for size in backend.batch_sizes)
        stats = (await client.get("/stats")).json()
        assert stats["index_version"] == 1
        assert stats["in_flight"] is False

        answers = await asyncio.gather(
            query(client, "q1"), query(client, "q2")
        )
        assert all(r.status_code == 200 for r in answers)
        assert backend.max_active == 1


@pytest.mark.asyncio
async def test_build_timeout_leaves_no_partial_index() -> None:
    backend = FakeBackend(delay=0.2)
    async with gateway_client(backend) as (_, client):
        response = await build(client, documents(8), timeout_ms=50)
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "request_timeout"
        # Let the abandoned backend batches drain.
        await asyncio.sleep(0.5)
        status = (await client.get("/v1/index/status")).json()
        assert status["indexed"] is False
        rejected = await query(client, "q")
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "index_unavailable"


@pytest.mark.asyncio
async def test_index_is_empty_after_restart_and_model_hint_is_checked() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        assert (await build(client, documents(4))).status_code == 200
        wrong_model = await client.post(
            "/v1/index/query",
            json={"query": "q", "model": "another-model"},
        )
        assert wrong_model.status_code == 409
        assert wrong_model.json()["error"]["code"] == "index_incompatible"

    # A fresh app (process restart) starts without an index.
    async with gateway_client(FakeBackend()) as (_, client):
        status = (await client.get("/v1/index/status")).json()
        assert status["indexed"] is False
        assert (await query(client, "q")).status_code == 503


@pytest.mark.asyncio
async def test_query_snapshot_is_pinned_before_body_parsing() -> None:
    """A rebuild while the query body is still streaming must not be used."""
    backend = FakeBackend()
    app = create_app(
        make_settings(), backend_transport=httpx.MockTransport(backend)
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            old_docs = documents(5, "old")
            assert (await build(client, old_docs)).status_code == 200

        # Stream a query whose body arrives only after a rebuild was accepted
        # and installed.  The snapshot was captured when the request was
        # accepted, before any body byte was read.
        body = json.dumps({"query": "old text 1", "timeout_ms": 5000}).encode()

        class Body:
            async def __aiter__(self):
                install = asyncio.create_task(build(client, documents(5, "new")))
                await asyncio.sleep(0.02)
                await install
                yield body

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/index/query",
                content=Body(),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["index_version"] == 1
            assert {m["text"] for m in payload["matches"]} <= {
                d["text"] for d in old_docs
            }

            status = (await client.get("/v1/index/status")).json()
            assert status["index_version"] == 2


@pytest.mark.asyncio
async def test_query_accepted_before_any_index_rejects_even_if_built_later() -> None:
    """An accepted query is bound to "no index"; a later build must not serve it."""
    backend = FakeBackend()
    app = create_app(
        make_settings(), backend_transport=httpx.MockTransport(backend)
    )
    async with app.router.lifespan_context(app):
        body = json.dumps({"query": "anything", "timeout_ms": 5000}).encode()

        class Body:
            async def __aiter__(self):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://gateway"
                ) as inner:
                    response = await inner.post(
                        "/v1/index/build",
                        json={"documents": documents(3)},
                    )
                    assert response.status_code == 200
                yield body

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/index/query",
                content=Body(),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 503, response.text
            assert response.json()["error"]["code"] == "index_unavailable"
            status = (await client.get("/v1/index/status")).json()
            assert status["index_version"] == 1


@pytest.mark.asyncio
async def test_query_dimension_mismatch_is_rejected() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        assert (await build(client, documents(4))).status_code == 200
        backend.dim = 6
        response = await query(client, "q")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "index_incompatible"


@pytest.mark.asyncio
async def test_index_endpoints_enforce_deadline_ceiling() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        response = await client.post(
            "/v1/index/build",
            json={"documents": documents(1), "timeout_ms": 600001},
        )
        assert response.status_code == 422
        response = await client.post(
            "/v1/index/query",
            json={"query": "q", "timeout_ms": 600001},
        )
        assert response.status_code == 422
        assert backend.calls == []


@pytest.mark.asyncio
async def test_concurrent_builds_are_serialized_and_timeout_does_not_wedge_lock() -> None:
    backend = FakeBackend(delay=0.08)
    async with gateway_client(backend) as (_, client):
        first = asyncio.create_task(build(client, documents(8, "first")))
        await asyncio.sleep(0.02)
        second = await build(client, documents(8, "second"), timeout_ms=40)
        assert second.status_code == 504
        assert second.json()["error"]["code"] == "request_timeout"

        assert (await first).status_code == 200
        status = (await client.get("/v1/index/status")).json()
        assert status["index_version"] == 1
        assert status["index_builds_failed"] == 1

        backend.delay = 0.0
        third = await build(client, documents(4, "third"))
        assert third.status_code == 200, third.text
        assert third.json()["index_version"] == 2
        assert backend.max_active == 1


@pytest.mark.asyncio
async def test_openapi_exposes_new_endpoints() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        schema = (await client.get("/openapi.json")).json()
        for path in ("/v1/index/build", "/v1/index/query", "/v1/index/status"):
            assert path in schema["paths"]
        docs = await client.get("/docs")
        assert docs.status_code == 200
