"""Errors shared by the gateway request and batching layers."""

from __future__ import annotations


class GatewayError(Exception):
    """Base class for failures that can be mapped to an HTTP response."""

    status_code = 500
    code = "internal_error"

    def __init__(
        self, message: str, *, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class RequestTimeout(GatewayError):
    """The request deadline (measured from acceptance) expired."""

    status_code = 504
    code = "request_timeout"


class ClientDisconnected(GatewayError):
    """The HTTP client went away before its embedding was delivered."""

    status_code = 499
    code = "client_disconnected"


class BackendError(GatewayError):
    """The single-slot embedding backend failed or returned invalid data."""

    status_code = 502
    code = "backend_error"


class ServiceUnavailable(GatewayError):
    """The gateway is shutting down and cannot serve this request."""

    status_code = 503
    code = "service_unavailable"


class ValidationError(GatewayError):
    """The request is well-formed JSON but violates gateway limits."""

    status_code = 422
    code = "validation_error"


class IndexUnavailable(GatewayError):
    """No corpus index exists yet, so a query cannot be served."""

    status_code = 503
    code = "index_unavailable"


class RevisionUnavailable(GatewayError):
    """The backend did not provide a model revision when one is required.

    A corpus index can only be built and queried when every inference result
    carries the revision that produced it; otherwise documents and queries
    from different model versions could be silently mixed.
    """

    status_code = 503
    code = "revision_unavailable"


class RevisionMismatch(GatewayError):
    """The backend's actual model revision differs from the pinned revision.

    For a build this means the backend switched revisions between acceptance
    and inference, so the build must be rejected and the old index kept.  For
    a query it means the query vector was produced by a different model
    revision than the snapshot index, so the index is unavailable for that
    vector -- old vectors or a newer index must never be substituted.
    """

    status_code = 409
    code = "revision_mismatch"


class IndexIncompatible(GatewayError):
    """The query vector does not fit the pinned index (e.g. dimension)."""

    status_code = 409
    code = "index_incompatible"
