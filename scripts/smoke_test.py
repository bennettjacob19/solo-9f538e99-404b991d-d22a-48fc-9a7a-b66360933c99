"""End-to-end smoke test against running gateway + mock backend.

Usage:
    python scripts/smoke_test.py [--gateway URL] [--backend URL]

Requires the compose stack (or anything exposing the same endpoints) to be
up.  Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def main(gateway: str, backend: str) -> int:
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        if not condition:
            failures += 1
        print(f"[{status}] {label}{(' :: ' + detail) if detail else ''}")

    async with httpx.AsyncClient(timeout=30) as client:
        for service, url in (("gateway", gateway), ("backend", backend)):
            health = await client.get(f"{url}/healthz")
            check(f"{service} healthz", health.status_code == 200)

        docs = await client.get(f"{gateway}/docs")
        check("gateway /docs reachable", docs.status_code == 200)

        # 1. Merge 4+4 texts into one batch and check index ordering.
        await client.post(f"{backend}/admin/reset")
        await client.post(f"{backend}/admin/config", json={"delay_ms": 250, "mode": "ok"})
        groups = [["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]]
        responses = await asyncio.gather(
            *[
                client.post(
                    f"{gateway}/v1/embeddings",
                    json={"input": texts, "timeout_ms": 5000},
                )
                for texts in groups
            ]
        )
        check("merged batch returns 200", all(r.status_code == 200 for r in responses))
        bodies = [r.json() for r in responses]
        check(
            "both callers share one backend batch",
            bodies[0]["batch_id"] == bodies[1]["batch_id"],
            f"{bodies[0]['batch_id']} vs {bodies[1]['batch_id']}",
        )
        check("batch size is 8", bodies[0]["batch_size"] == 8, str(bodies[0]["batch_size"]))
        check(
            "indices preserved per caller",
            all(
                [item["index"] for item in body["data"]] == list(range(4))
                for body in bodies
            ),
        )
        stats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend saw one batch of 8",
            [len(entry["texts"]) for entry in stats["requests"]] == [8],
            str([len(entry["texts"]) for entry in stats["requests"]]),
        )
        check(
            "single-slot backend never ran two batches concurrently",
            stats["max_concurrency"] == 1 and stats["concurrency_violations"] == 0,
            f"max={stats['max_concurrency']} violations={stats['concurrency_violations']}",
        )

        # 2. Deadline expiry: late vectors are discarded, next batch waits.
        await client.post(f"{backend}/admin/reset")
        await client.post(f"{backend}/admin/config", json={"delay_ms": 500, "mode": "ok"})
        expired = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["slow"], "timeout_ms": 80},
        )
        check("expired request returns 504", expired.status_code == 504, expired.text)
        started = time.monotonic()
        nxt = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["next"], "timeout_ms": 5000},
        )
        waited = time.monotonic() - started
        check("follow-up request succeeds", nxt.status_code == 200, nxt.text)
        check(
            "follow-up waited for the abandoned batch",
            waited >= 0.3,
            f"waited {waited:.2f}s",
        )
        gstats = (await client.get(f"{gateway}/stats")).json()
        check("late vectors not counted as completed", gstats["completed_requests"] >= 1)
        check("no stuck requests", gstats["pending_requests"] == 0 and not gstats["in_flight"])
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend batches never overlap",
            bstats["max_concurrency"] == 1 and bstats["concurrency_violations"] == 0,
            f"max={bstats['max_concurrency']}",
        )

        # 3. Backend failure fails every waiter, then service recovers.
        await client.post(f"{backend}/admin/reset")
        await client.post(
            f"{backend}/admin/config", json={"delay_ms": 50, "mode": "error"}
        )
        failed = await asyncio.gather(
            *[
                client.post(
                    f"{gateway}/v1/embeddings",
                    json={"input": texts, "timeout_ms": 5000},
                )
                for texts in (["f1"], ["f2", "f3"])
            ]
        )
        check(
            "whole batch fails with 502",
            all(r.status_code == 502 for r in failed),
            str([r.status_code for r in failed]),
        )
        await client.post(f"{backend}/admin/config", json={"mode": "ok"})
        recovered = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["after"], "timeout_ms": 5000}
        )
        check("requests still served after backend error", recovered.status_code == 200)
        gstats = (await client.get(f"{gateway}/stats")).json()
        check(
            "no slot leak after error",
            gstats["pending_requests"] == 0 and not gstats["in_flight"],
            json.dumps(
                {
                    "pending": gstats["pending_requests"],
                    "in_flight": gstats["in_flight"],
                }
            ),
        )

        # 4. Validation limits.
        too_many = await client.post(
            f"{gateway}/v1/embeddings", json={"input": list("abcde")}
        )
        check("5 texts rejected", too_many.status_code == 422, str(too_many.status_code))
        zero_timeout = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["x"], "timeout_ms": 0}
        )
        check("timeout_ms=0 rejected", zero_timeout.status_code == 422)

        # 5. Corpus index: query without an index is explicitly rejected.
        await client.post(f"{backend}/admin/config", json={"delay_ms": 0, "mode": "ok"})
        await client.post(f"{backend}/admin/revision", json={"revision": "smoke-r1"})
        no_index = await client.post(
            f"{gateway}/v1/index/query", json={"query": "anything"}
        )
        check(
            "query without index -> index_unavailable",
            no_index.status_code == 503
            and no_index.json().get("error", {}).get("code") == "index_unavailable",
            no_index.text,
        )

        # 6. Build a 20-document corpus (multiple batches) and query it.
        await client.post(f"{backend}/admin/reset")
        await client.post(f"{backend}/admin/config", json={"delay_ms": 100})
        docs = [
            {"id": f"doc-{i}", "text": f"smoke document number {i}"}
            for i in range(20)
        ]
        built = await client.post(
            f"{gateway}/v1/index/build",
            json={"documents": docs, "timeout_ms": 10000},
        )
        check("corpus build returns 200", built.status_code == 200, built.text)
        build_body = built.json()
        check(
            "build pins the current revision",
            build_body.get("model_revision") == "smoke-r1",
            str(build_body.get("model_revision")),
        )
        check(
            "build reports all 20 documents",
            build_body.get("document_count") == 20,
            str(build_body.get("document_count")),
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "build went through batches of at most 8, single-slot",
            all(len(entry["texts"]) <= 8 for entry in bstats["requests"])
            and bstats["max_concurrency"] == 1,
            str([len(entry["texts"]) for entry in bstats["requests"]]),
        )

        answered = await client.post(
            f"{gateway}/v1/index/query",
            json={"query": "smoke document number 7", "timeout_ms": 5000},
        )
        check("query returns 200", answered.status_code == 200, answered.text)
        answer = answered.json()
        check(
            "query returns at most 3 matches",
            0 < len(answer.get("matches", [])) <= 3,
            str(len(answer.get("matches", []))),
        )
        check(
            "query reports index version and revision",
            answer.get("index_version") == build_body["index_version"]
            and answer.get("model_revision") == "smoke-r1",
            json.dumps(
                {
                    "version": answer.get("index_version"),
                    "revision": answer.get("model_revision"),
                }
            ),
        )
        top = answer["matches"][0]["id"]
        check(
            "most similar document ranks first",
            top == "doc-7",
            f"top={top}",
        )

        # 7. Switching model revision makes the old index unavailable for new
        #    queries; a rebuild installs a new version.
        await client.post(f"{backend}/admin/revision", json={"revision": "smoke-r2"})
        mismatched = await client.post(
            f"{gateway}/v1/index/query",
            json={"query": "smoke document number 7", "timeout_ms": 5000},
        )
        check(
            "query under a different revision -> revision_mismatch",
            mismatched.status_code == 409
            and mismatched.json().get("error", {}).get("code")
            == "revision_mismatch",
            mismatched.text,
        )
        old = await client.get(f"{gateway}/v1/index/status")
        check(
            "old index is still the installed version",
            old.json().get("model_revision") == "smoke-r1"
            and old.json().get("index_version") == build_body["index_version"],
            old.text,
        )

        await client.post(f"{backend}/admin/config", json={"mode": "error"})
        broken = await client.post(
            f"{gateway}/v1/index/build",
            json={"documents": docs, "timeout_ms": 5000},
        )
        check(
            "failed rebuild keeps the old index",
            broken.status_code == 502,
            broken.text,
        )
        await client.post(f"{backend}/admin/config", json={"mode": "ok"})
        still_old = (await client.get(f"{gateway}/v1/index/status")).json()
        check(
            "old index unchanged after failed rebuild",
            still_old.get("index_version") == build_body["index_version"]
            and still_old.get("model_revision") == "smoke-r1",
            json.dumps(still_old),
        )

        rebuilt = await client.post(
            f"{gateway}/v1/index/build",
            json={"documents": docs, "timeout_ms": 10000},
        )
        check("rebuild under new revision returns 200", rebuilt.status_code == 200, rebuilt.text)
        check(
            "rebuild installs a new version on the new revision",
            rebuilt.json().get("index_version") == build_body["index_version"] + 1
            and rebuilt.json().get("model_revision") == "smoke-r2",
            rebuilt.text,
        )
        new_answer = await client.post(
            f"{gateway}/v1/index/query",
            json={"query": "smoke document number 7", "timeout_ms": 5000},
        )
        check(
            "query works again on the new index version",
            new_answer.status_code == 200
            and new_answer.json().get("index_version")
            == rebuilt.json().get("index_version"),
            new_answer.text,
        )

    print()
    print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", default="http://127.0.0.1:9000")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.gateway, args.backend)))
