"""Mock backend revision behaviour: current revision, pinned inference."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.app.main import app as backend_app
from backend.app.main import make_embedding, state


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state, "revision", "r1")
    monkeypatch.setattr(state, "revision_switches", 0)
    monkeypatch.setattr(state.config, "delay_ms", 0)
    monkeypatch.setattr(state.config, "mode", "ok")
    state.requests = []


async def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend_app),
        base_url="http://backend",
    )


@pytest.mark.asyncio
async def test_revision_endpoint_exposes_only_current_revision() -> None:
    async with await _client() as client:
        assert (await client.get("/v1/model/revision")).json() == {"revision": "r1"}
        switched = await client.post(
            "/admin/revision", json={"revision": "r2"}
        )
        assert switched.status_code == 200
        assert (await client.get("/v1/model/revision")).json() == {"revision": "r2"}
        # No history endpoint exists for old revisions.
        assert (await client.get("/v1/model/revision/r1")).status_code == 404


@pytest.mark.asyncio
async def test_revisions_share_dimension_but_change_vectors() -> None:
    async with await _client() as client:
        first = (
            await client.post("/v1/embeddings", json={"input": ["same text"]})
        ).json()
        assert first["revision"] == "r1"
        await client.post("/admin/revision", json={"revision": "r2"})
        second = (
            await client.post("/v1/embeddings", json={"input": ["same text"]})
        ).json()
        assert second["revision"] == "r2"
        assert len(first["data"][0]["embedding"]) == len(
            second["data"][0]["embedding"]
        )
        assert first["data"][0]["embedding"] != second["data"][0]["embedding"]
        assert make_embedding("same text", 8, "r1") != make_embedding(
            "same text", 8, "r2"
        )


@pytest.mark.asyncio
async def test_switch_during_inference_does_not_rewrite_started_result() -> None:
    state.config.delay_ms = 150
    async with await _client() as client:
        inference = asyncio.create_task(
            client.post("/v1/embeddings", json={"input": ["in-flight text"]})
        )
        # Wait until the request has entered the backend, then switch.
        for _ in range(50):
            if state.requests:
                break
            await asyncio.sleep(0.01)
        switched = await client.post("/admin/revision", json={"revision": "r2"})
        assert switched.status_code == 200

        body = (await inference).json()
        assert body["revision"] == "r1"
        assert body["data"][0]["embedding"] == make_embedding(
            "in-flight text", state.dim, "r1"
        )
        # Subsequent inference uses the new revision.
        later = (
            await client.post("/v1/embeddings", json={"input": ["in-flight text"]})
        ).json()
        assert later["revision"] == "r2"
        assert later["data"][0]["embedding"] != body["data"][0]["embedding"]


@pytest.mark.asyncio
async def test_blank_revision_is_rejected() -> None:
    async with await _client() as client:
        response = await client.post("/admin/revision", json={"revision": "   "})
        assert response.status_code == 400
        assert state.revision == "r1"
