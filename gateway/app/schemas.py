"""API schemas for the gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingRequest(BaseModel):
    """One gateway request carrying 1..4 short texts.

    ``input`` follows the OpenAI embeddings shape (a string or a list of
    strings). ``texts`` is accepted as a convenience alias.
    """

    input: str | list[str] = Field(
        ...,
        description="One to four short texts (a lone string is also allowed).",
        examples=[["hello world", "second text"]],
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        description=(
            "End-to-end deadline in milliseconds, starting when the gateway "
            "accepts the request. It covers queueing plus the backend call."
        ),
        examples=[1000],
    )
    model: str | None = Field(
        default=None,
        description="Optional model hint forwarded to the backend.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_texts_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "input" not in data and "texts" in data:
                data = {**data, "input": data["texts"]}
            if "input" not in data and "prompt" in data:
                data = {**data, "input": data["prompt"]}
        return data

    @field_validator("input")
    @classmethod
    def _normalize_input(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            return value
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("input must be a string or a list of strings")
        return value

    def as_texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    batch_id: str = Field(description="Identifier shared by one backend batch.")
    batch_size: int = Field(
        description="Total number of texts sent to the backend in this batch."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class CorpusDocumentIn(BaseModel):
    id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Caller-assigned document id; unique within one build.",
        examples=["doc-1"],
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Text to embed and keep in the in-memory index.",
        examples=["the quick brown fox"],
    )


class IndexBuildRequest(BaseModel):
    """Submit a whole corpus; it replaces the old index only if complete."""

    documents: list[CorpusDocumentIn] = Field(
        ...,
        min_length=1,
        max_length=32,
        description=(
            "The complete corpus, 1-32 documents with unique ids. The build "
            "is all-or-nothing: any missing vector or revision mismatch "
            "keeps the previous index."
        ),
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        description=(
            "End-to-end deadline in milliseconds, starting when the gateway "
            "accepts the request. Covers the revision lookup plus all "
            "embedding batches."
        ),
        examples=[5000],
    )
    model: str | None = Field(
        default=None,
        description="Optional model hint forwarded to the backend.",
    )


class IndexBuildResponse(BaseModel):
    index_version: int = Field(description="Newly installed index version.")
    model_revision: str = Field(
        description="Backend revision pinned at acceptance; every document "
        "vector was produced by this revision."
    )
    model: str
    dimension: int
    document_count: int
    batch_ids: list[str] = Field(
        description="Backend batches that produced the document vectors."
    )


class IndexQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Text to embed and compare against the current index.",
        examples=["quick brown fox"],
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        description=(
            "End-to-end deadline in milliseconds, starting when the gateway "
            "accepts the request."
        ),
        examples=[1000],
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional model hint. If given it must match the model used to "
            "build the index."
        ),
    )


class QueryMatch(BaseModel):
    id: str
    text: str
    score: float = Field(description="Cosine similarity, rounded to 6 places.")


class IndexQueryResponse(BaseModel):
    index_version: int = Field(
        description="Version of the index snapshot this query was accepted "
        "against; a concurrent rebuild cannot change it."
    )
    model_revision: str = Field(
        description="Revision the index (and this query embedding) use."
    )
    model: str
    matches: list[QueryMatch] = Field(
        description="Up to 3 documents with the highest cosine similarity."
    )
    batch_id: str
    batch_size: int


class IndexStatusResponse(BaseModel):
    indexed: bool
    index_version: int | None = None
    document_count: int | None = None
    model_revision: str | None = None
    model: str | None = None
    dimension: int | None = None
    created_ago_seconds: float | None = None
    index_builds_completed: int
    index_builds_failed: int
    index_queries_completed: int
    index_queries_rejected: int


class StatsResponse(BaseModel):
    accepted_requests: int
    completed_requests: int
    timed_out_requests: int
    disconnected_requests: int
    failed_requests: int
    batches_dispatched: int
    backend_errors: int
    backend_texts_sent: int
    pending_requests: int
    in_flight: bool
    last_batch: dict[str, Any] | None = None
    index_builds_completed: int = 0
    index_builds_failed: int = 0
    index_queries_completed: int = 0
    index_queries_rejected: int = 0
    index_version: int | None = None
    index_documents: int | None = None
    index_revision: str | None = None
