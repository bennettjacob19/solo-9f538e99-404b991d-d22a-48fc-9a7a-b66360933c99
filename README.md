# Embedding Batch Gateway

A small FastAPI service that accepts **1–4 short texts per HTTP call** and
merges concurrent calls into **single-flight batches of at most 8 texts** for a
single-slot CPU embedding backend. It ships with a controllable local mock
backend and a Docker Compose file that starts both.

On top of the batch endpoint it provides an **in-memory corpus index**: submit
a whole corpus of up to 32 uniquely identified texts, then query it for the
three most similar documents. Builds and queries are pinned to the backend
model revision and use the same batching scheduler as normal embeddings, so
vectors from two model revisions can never be mixed in one index.

```
client ──► gateway :8000 ──(one batch at a time)──► backend :9000
             │ batching, deadlines, ordering           │ configurable delay
             │ single-flight, error fan-out            │ error injection
             │ corpus index + revision pinning          │ revisions r1, r2, …
             └ /docs, /stats, /v1/index/*              └ /docs, /admin/*
```

## Quick start

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| Gateway API docs (interactive) | http://localhost:8000/docs |
| Gateway stats | http://localhost:8000/stats |
| Gateway health | http://localhost:8000/healthz |
| Mock backend docs | http://localhost:9000/docs |
| Mock backend stats (real concurrency) | http://localhost:9000/admin/stats |

No Docker? The same two apps run directly:

```bash
pip install -r requirements-dev.txt

uvicorn backend.app.main:app --port 9000 &
EMBEDDING_BACKEND_URL=http://127.0.0.1:9000/v1/embeddings \
EMBEDDING_BACKEND_REVISION_URL=http://127.0.0.1:9000/v1/model/revision \
  uvicorn gateway.app.main:app --port 8000 &
```

Run the tests and the live smoke test:

```bash
pytest -q                  # no network required
python scripts/smoke_test.py   # requires both services running
```

## API

### `POST /v1/embeddings`

```json
{
  "input": ["first text", "second text"],
  "timeout_ms": 1000,
  "model": "local-demo"
}
```

* `input` — a string or a list of 1–4 strings. `texts` is accepted as an alias.
* `timeout_ms` — end-to-end deadline in milliseconds, **measured from the
  moment the gateway accepts the request** (a middleware stamps it before the
  body is parsed). Defaults to `GATEWAY_DEFAULT_TIMEOUT_MS` (5000) when omitted.
* `model` — optional; forwarded to the backend, otherwise `EMBEDDING_MODEL`.

Response (indices are always `0..n-1` in the caller's original input order):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
  ],
  "model": "local-demo",
  "batch_id": "batch-7",
  "batch_size": 8
}
```

`batch_id`/`batch_size` describe the backend batch the caller was merged into
(useful for demos and tests).

| Status | `error.code` | Meaning |
| --- | --- | --- |
| 422 | `validation_error` | text count, `timeout_ms` or schema violation |
| 502 | `backend_error` | backend HTTP error, timeout, or invalid payload |
| 503 | `service_unavailable` | gateway is shutting down |
| 504 | `request_timeout` | deadline hit while queueing or waiting |

### Corpus index

The index lives in gateway memory only (one process) and is intentionally lost
on restart. It holds **one immutable snapshot** at a time.

#### `POST /v1/index/build`

Submit the **whole corpus** in one call. Up to 32 documents, each with a
unique `id`; any duplicate/blank id or out-of-range size is rejected with
`422` before the backend is called.

```json
{
  "documents": [
    {"id": "doc-1", "text": "the quick brown fox"},
    {"id": "doc-2", "text": "a lazy dog"}
  ],
  "timeout_ms": 5000
}
```

* The backend's **current model revision is pinned when the build is
  accepted, before any document embedding is scheduled** (metadata-only `GET`
  to `EMBEDDING_BACKEND_REVISION_URL`; it does not occupy the inference slot).
  Concurrent builds are serialized; one that cannot start before its deadline
  returns `504` without touching the installed index.
* Document embeddings are sent through the normal single-flight scheduler in
  batches of at most 8. A 32-document build therefore becomes at most four
  backend batches, and it never runs concurrently with ordinary embedding
  calls.
* The new index replaces the old one **atomically, and only if** every
  document has a vector, all vectors share one dimension, and every
  backend-reported revision equals the pinned revision. Any failure
  (backend error, timeout, incomplete response, revision change mid-build)
  returns an error and **keeps the previous index** unchanged.
* A rebuild does not block queries: until the swap, queries keep reading the
  old snapshot.

Response:

```json
{
  "index_version": 1,
  "model_revision": "r1",
  "model": "local-demo",
  "dimension": 8,
  "document_count": 2,
  "batch_ids": ["batch-6"]
}
```

#### `POST /v1/index/query`

```json
{"query": "quick fox", "timeout_ms": 1000}
```

* The gateway captures the current index snapshot (content **and** revision)
  at HTTP acceptance, before the request body is parsed. A build finishing
  while the request body streams or while the query waits for inference
  cannot change what this query searches — including the case where no index
  existed at acceptance: that query is rejected rather than served by the
  newly built index.
* The query embedding is produced through the same scheduler. Its
  backend-reported revision must equal the snapshot revision; otherwise the
  request fails with `409 revision_mismatch` and details of both revisions.
  The gateway never substitutes the old vectors or a newer index.
* Returns the **3 documents with the highest cosine similarity**, highest
  first (document order breaks ties), plus the index version used:

```json
{
  "index_version": 1,
  "model_revision": "r1",
  "model": "local-demo",
  "matches": [
    {"id": "doc-1", "text": "the quick brown fox", "score": 0.97},
    {"id": "doc-2", "text": "a lazy dog", "score": 0.42}
  ],
  "batch_id": "batch-7",
  "batch_size": 1
}
```

Query errors:

| Status | `error.code` | Meaning |
| --- | --- | --- |
| 409 | `index_incompatible` | dimension differs from the index, or a `model` hint does not match the indexed model |
| 409 | `revision_mismatch` | the query vector's actual revision differs from the pinned index revision |
| 503 | `index_unavailable` | no index exists yet (including after a restart) |
| 503 | `revision_unavailable` | the backend failed to report the revision needed to pin/compare |
| 422 / 504 / 502 | as above | validation, deadline, backend failure |

Build errors reuse these codes: `502 backend_error`, `504 request_timeout`,
`409 revision_mismatch` (revision changed during the build), and
`503 revision_unavailable` (backend cannot report revisions, or
`EMBEDDING_BACKEND_REVISION_URL` is unset).

#### `GET /v1/index/status`

Shows `indexed`, `index_version`, `document_count`, `model_revision`, `model`,
`dimension`, `created_ago_seconds`, and build/query counters. `GET /stats`
includes the same counters plus the scheduler stats.

### Ops endpoints

* `GET /healthz` — liveness.
* `GET /stats` — counters (`accepted`, `completed`, `timed_out`,
  `disconnected`, `failed`, `batches_dispatched`, `backend_errors`),
  `pending_requests`, `in_flight`, `last_batch`, and index counters/state.

## Batching semantics

1. **Merge window.** When the backend is idle, the first pending request opens
   a `GATEWAY_BATCH_WINDOW_MS` (20 ms) window starting when it became pending.
   Requests arriving inside the window join the batch FIFO.
2. **Size limit.** A batch never exceeds `GATEWAY_MAX_TEXTS_PER_BATCH` (8)
   texts. Requests are never split; a request only joins if it fits whole.
3. **No re-arming.** Requests that piled up while the backend was busy already
   elapsed their window, so they are merged and dispatched immediately once the
   backend frees up — no extra 20 ms per batch.
4. **Single-flight.** Exactly one dispatcher task exists, so at most one
   backend batch is in flight. The next batch is not sent until the previous
   response *or error* has been fully received and processed — even if every
   caller of that batch already timed out or disconnected.
5. **Deadline/abandon.** A request whose deadline expires or whose client
   disconnects while **pending** is removed from the queue and never sent. A
   request abandoned while **in flight** keeps its slot occupied until the
   backend answers, but its late vectors are discarded, never delivered to the
   expired caller.
6. **Errors.** Any backend failure (non-2xx, timeout, malformed body, wrong
   count, duplicate/out-of-range index, non-numeric vector) fails **all**
   waiters of that batch with `502 backend_error`, releases the slot in a
   `finally` block, and leaves the gateway ready for the next request.

Disconnect detection polls `Request.is_disconnected()` every
`GATEWAY_DISCONNECT_POLL_INTERVAL_MS`. On ASGI servers that cannot expose the
receive channel, probing is disabled rather than treated as a disconnect; the
deadline still bounds the caller's wait.

## Configuration

Gateway (environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND_URL` | `http://127.0.0.1:9000/v1/embeddings` | backend endpoint |
| `EMBEDDING_BACKEND_REVISION_URL` | *unset* | endpoint returning `{"revision": "..."}`; required for index builds/queries |
| `EMBEDDING_MODEL` | `local-demo` | default model name |
| `EMBEDDING_API_KEY` | *unset* | sent as `Authorization: Bearer …` |
| `EMBEDDING_BACKEND_TIMEOUT_SECONDS` | `30` | whole-request backend timeout |
| `EMBEDDING_BACKEND_CONNECT_TIMEOUT_SECONDS` | `5` | connect timeout |
| `GATEWAY_BATCH_WINDOW_MS` | `20` | idle merge window |
| `GATEWAY_MAX_TEXTS_PER_REQUEST` | `4` | max texts per HTTP call |
| `GATEWAY_MAX_TEXTS_PER_BATCH` | `8` | max texts per backend batch |
| `GATEWAY_DEFAULT_TIMEOUT_MS` | `5000` | default deadline |
| `GATEWAY_MAX_TIMEOUT_MS` | `600000` | max accepted deadline |
| `GATEWAY_DISCONNECT_POLL_INTERVAL_MS` | `20` | disconnect poll cadence |

Mock backend (also editable live via `POST /admin/config`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_DELAY_MS` | `300` in compose | simulated inference latency |
| `MOCK_ERROR_RATE` | `0.0` | random failure probability |
| `MOCK_ERROR_STATUS` | `503` | status used for injected errors |
| `MOCK_MODE` | `ok` | `ok`, `error`, `malformed`, `short`, `duplicate_index`, `shuffle` |
| `MOCK_SHUFFLE_RESPONSE` | `false` | deliberately reverse `data` order |
| `MOCK_MAX_INPUTS` | `8` | returns 400 for larger batches |
| `MOCK_DIM` | `8` | embedding dimension |
| `MOCK_REVISION` | `r1` | initial model revision |

The mock exposes `GET /v1/model/revision` (current revision only — no
history) and `POST /admin/revision {"revision": "r2"}` to switch it. Different
revisions use the **same dimension but different vectors**; each inference
pins the revision when it starts and returns it in the `revision` field, so a
switch during the simulated delay never rewrites an in-flight result.

The mock logs every request and tracks `max_concurrency`,
`concurrency_violations`, `oversized_batches`, `max_batch_seen` and a request
log at `GET /admin/stats`. Because it is single-slot in intent, any overlap
above 1 (or any batch > `max_inputs`) is recorded as a violation.

## Verifying the key paths

```bash
# Merge + order
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["a","b"],"timeout_ms":1000}' | jq
curl -s localhost:9000/admin/stats | jq '.max_concurrency, .max_batch_seen'

# Inject a slow backend and expire a request
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"delay_ms":800,"mode":"ok"}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' -d '{"input":["slow"],"timeout_ms":100}'

# Inject an error for the next batch
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"error"}'
```

### Demonstrating the index and a revision upgrade

```bash
# 1. Query before any index exists -> explicit 503 index_unavailable
curl -s localhost:8000/v1/index/query -H 'content-type: application/json' \
  -d '{"query":"fox"}'

# 2. Build a corpus (20 docs => 3 backend batches of <=8), pinned to r1
curl -s localhost:8000/v1/index/build -H 'content-type: application/json' \
  -d '{"documents":[
        {"id":"d1","text":"the quick brown fox"},
        {"id":"d2","text":"a lazy dog"},
        {"id":"d3","text":"quick foxes jump"}]}'

# 3. Query it
curl -s localhost:8000/v1/index/query -H 'content-type: application/json' \
  -d '{"query":"quick fox"}' | jq '.index_version, .model_revision, .matches'

# 4. Upgrade the mock model revision (same dimension, different vectors)
curl -s -X POST localhost:9000/admin/revision \
  -H 'content-type: application/json' -d '{"revision":"r2"}'

# 5. Old index is now unavailable for queries -> 409 revision_mismatch
curl -s localhost:8000/v1/index/query -H 'content-type: application/json' \
  -d '{"query":"quick fox"}'

# 6. Rebuild under r2 -> version 2; queries work again
curl -s localhost:8000/v1/index/build -H 'content-type: application/json' \
  -d '{"documents":[{"id":"d1","text":"the quick brown fox"},
                    {"id":"d2","text":"a lazy dog"}]}'
curl -s localhost:8000/v1/index/query -H 'content-type: application/json' \
  -d '{"query":"quick fox"}'

# 7. A failing rebuild leaves the installed index untouched
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"error"}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/index/build \
  -H 'content-type: application/json' \
  -d '{"documents":[{"id":"d9","text":"will not be installed"}]}'
curl -s localhost:8000/v1/index/status | jq '.index_version, .model_revision'
```

# Verification summary

`pytest -q` covers merge sizing (never > 8), request-order preservation,
window timing, pending-expiry removal, single-flight after every caller left,
late-vector discard, error fan-out with recovery, malformed backend payloads,
disconnect abandonment, plus the index semantics: build/query through the
scheduler, unique-id and 32-document limits, atomic replacement, failed-rebuild
rollback, revision pinning at acceptance, mid-build revision change rejection,
revision-mismatched query rejection without substitution, query snapshot
pinning across a concurrent rebuild, missing index rejection, model/dimension
checks, and restart clearing the index. `scripts/smoke_test.py` repeats the key
paths against real uvicorn processes, including a full r1 → r2 revision switch
and rebuild, and checks the mock backend's recorded
`max_concurrency`/`concurrency_violations`.

## Project layout

```
gateway/app/config.py    settings from env
gateway/app/batcher.py   single-flight queue, merge window, deadlines
gateway/app/corpus.py    revision-pinned in-memory corpus index
gateway/app/main.py      FastAPI app, routes, acceptance-time middleware
gateway/app/schemas.py   request/response models
backend/app/main.py      controllable mock embedding backend + revisions
tests/                   batching, window, timeout, disconnect, index tests
scripts/smoke_test.py    live end-to-end checks against running services
docker-compose.yml       gateway + mock backend with healthchecks
```

## Known limitations

* The deadline is enforced by the gateway only; the in-flight backend call is
  not aborted on caller timeout because the batch must still be drained before
  the next one is sent.
* `batch_size`/`batch_id` are informational; clients should rely on `index`
  ordering only.
* Disconnect-abandonment is best effort and depends on the ASGI server
  surfacing `http.disconnect` (verified with uvicorn).
* Batching state is per process. Keep one uvicorn worker (as in the Docker
  image) unless each replica has its own backend slot; otherwise two workers
  can issue one batch each concurrently.
* The corpus index is in memory only: it is per process, lost on restart, and
  one gateway process holds at most one index snapshot.
* Index builds are serialized among themselves; a build holds no lock while a
  query runs, so queries never wait for a rebuild to finish.
* A build pins the revision returned by the configured revision endpoint. If
  the backend switches revisions between the pin and the first document
  batch, the build fails and the old index is kept — this is intentional, but
  it means an operator switching the model should rebuild afterwards.
* The mock backend is for local verification only — no auth, no persistence.
