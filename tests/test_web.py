from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from pawahara_harness.context import BeamCandidate, ContextStore, ThoughtSeed
from pawahara_harness.web import AgentMonitorServer, build_monitor_snapshot, read_monitor_file, render_monitor_page


def test_monitor_snapshot_exposes_worker_files_and_details(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("solve the challenge")
    prompt_path = store.write_prompt(run, "workers", "d0_w0_test", "worker prompt")
    store.append_event(
        run,
        "worker.started",
        {
            "candidate": "d0_w0_test",
            "depth": 0,
            "index": 0,
            "parent": None,
            "seed": {
                "id": "seed-1",
                "label": "instrumentation",
                "instruction": "measure first",
                "novelty_targets": ["observability"],
            },
            "prompt_path": str(prompt_path),
        },
    )
    response_path = store.write_response(run, "workers", "d0_w0_test", "worker stdout text")
    candidate = BeamCandidate(
        id="d0_w0_test",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(
            id="seed-1",
            label="instrumentation",
            instruction="measure first",
            novelty_targets=("observability",),
        ),
        score=0.7,
        status="promising",
        summary="found a faster path",
        next_context="keep benchmark result",
        prompt_path=str(prompt_path),
        response_path=str(response_path),
    )
    store.write_candidate(run, candidate)
    store.append_event(
        run,
        "worker.invocation",
        {
            "candidate": candidate.id,
            "exit_code": 0,
            "sandbox_id": "local",
            "command": "codex exec",
        },
    )

    snapshot = build_monitor_snapshot(store.runs_dir)
    worker = next(node for node in snapshot["nodes"] if node["id"] == "worker:d0_w0_test")

    assert snapshot["run"]["run_id"] == run.run_id
    assert snapshot["counts"]["candidates"] == 1
    assert worker["status"] == "promising"
    assert worker["details"]["candidate"]["summary"] == "found a faster path"
    assert worker["details"]["invocation"]["exit_code"] == 0
    assert any(file["label"] == "prompt" and "worker prompt" in file["preview"] for file in worker["files"])
    assert any(file["label"] == "response" and "worker stdout text" in file["preview"] for file in worker["files"])
    assert any(file["relative_path"] == "events.jsonl" for file in snapshot["files"])


def test_monitor_file_api_is_restricted_to_runs_dir(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("inspect files")
    response_path = store.write_response(run, "workers", "worker", "full response")

    payload, status = read_monitor_file(store.runs_dir, str(response_path))
    assert status == 200
    assert payload["content"] == "full response"

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    blocked, blocked_status = read_monitor_file(store.runs_dir, str(outside))
    assert blocked_status == 404
    assert not blocked["ok"]


def test_monitor_server_serves_snapshot_and_files(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("serve monitor")
    response_path = store.write_response(run, "workers", "worker", "served response")
    server = AgentMonitorServer(store.runs_dir, port=0).start()
    try:
        with urlopen(server.url + "api/latest", timeout=2) as handle:
            snapshot = json.loads(handle.read().decode("utf-8"))
        with urlopen(server.url + "api/file?" + urlencode({"path": str(response_path)}), timeout=2) as handle:
            file_payload = json.loads(handle.read().decode("utf-8"))
    finally:
        server.stop()

    assert snapshot["run"]["run_id"] == run.run_id
    assert file_payload["content"] == "served response"


def test_monitor_page_renders_state_as_trees() -> None:
    page = render_monitor_page()

    assert "agentTree" in page
    assert "eventTree" in page
    assert "renderAgentForest" in page
    assert "objectTree" in page
    assert "renderFileTree" in page
