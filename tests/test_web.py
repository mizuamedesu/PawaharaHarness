from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from pawahara_harness.context import BeamCandidate, ContextStore, ThoughtSeed
from pawahara_harness.web import (
    AgentMonitorServer,
    build_monitor_snapshot,
    read_candidate_files_with_previews,
    read_monitor_file,
    render_monitor_page,
)


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


def test_monitor_snapshot_refreshes_rewritten_same_size_file_preview(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("inspect rewritten previews")
    prompt_path = store.write_prompt(run, "workers", "d0_w0_test", "worker prompt")
    response_path = store.write_response(run, "workers", "d0_w0_test", "AAAA")
    fixed_ns = 1_700_000_000_123_456_789
    os.utime(response_path, ns=(fixed_ns, fixed_ns))
    candidate = BeamCandidate(
        id="d0_w0_test",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-1", label="cache", instruction="measure previews"),
        score=0.7,
        status="promising",
        summary="first",
        next_context="next",
        prompt_path=str(prompt_path),
        response_path=str(response_path),
    )
    store.write_candidate(run, candidate)

    first = build_monitor_snapshot(store.runs_dir)
    response_path.write_text("BBBB", encoding="utf-8")
    os.utime(response_path, ns=(fixed_ns, fixed_ns))
    second = build_monitor_snapshot(store.runs_dir)

    def response_preview(snapshot: dict[str, object]) -> str:
        worker = next(node for node in snapshot["nodes"] if node["id"] == "worker:d0_w0_test")  # type: ignore[index]
        return next(file["preview"] for file in worker["files"] if file["path"] == str(response_path))  # type: ignore[index]

    assert response_preview(first) == "AAAA"
    assert response_preview(second) == "BBBB"


def test_monitor_snapshot_skips_missing_candidate_file_paths(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("inspect missing files")
    prompt_path = store.write_prompt(run, "workers", "d0_w0_test", "worker prompt")
    missing_response = Path(run.root_dir) / "workers" / "missing.response.txt"
    missing_artifact = Path(run.root_dir) / "workers" / "missing-artifact.txt"
    candidate = BeamCandidate(
        id="d0_w0_test",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-1", label="missing", instruction="missing files"),
        score=0.1,
        status="dead_end",
        summary="missing file case",
        next_context="next",
        prompt_path=str(prompt_path),
        response_path=str(missing_response),
        artifacts=(str(missing_artifact),),
    )
    store.write_candidate(run, candidate)

    snapshot = build_monitor_snapshot(store.runs_dir)
    worker = next(node for node in snapshot["nodes"] if node["id"] == "worker:d0_w0_test")

    assert worker["status"] == "dead_end"
    assert all(file["path"] != str(missing_response) for file in worker["files"])
    assert all(file["path"] != str(missing_artifact) for file in worker["files"])


def test_candidate_file_cache_refreshes_and_returns_isolated_data(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / "d0_w0.json"
    candidate_path.write_text(
        json.dumps({"id": "d0_w0", "summary": "first", "seed": {"label": "cache"}}),
        encoding="utf-8",
    )

    first_candidates, first_previews = read_candidate_files_with_previews(candidate_dir)
    first_candidates[0]["summary"] = "mutated"
    second_candidates, _second_previews = read_candidate_files_with_previews(candidate_dir)
    candidate_path.write_text(
        json.dumps({"id": "d0_w0", "summary": "second value", "seed": {"label": "cache"}}),
        encoding="utf-8",
    )
    third_candidates, third_previews = read_candidate_files_with_previews(candidate_dir)

    assert first_previews[str(candidate_path)]
    assert second_candidates[0]["summary"] == "first"
    assert third_candidates[0]["summary"] == "second value"
    assert "second value" in third_previews[str(candidate_path)]


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


def test_monitor_snapshot_treats_orphan_prompt_only_worker_as_interrupted(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("continue the work")
    store.write_json(Path(run.root_dir) / "result.json", {"previous": "finished"})
    prompt_path = store.write_prompt(run, "workers", "d1_w0_running", "still working")

    snapshot = build_monitor_snapshot(store.runs_dir)
    worker = next(node for node in snapshot["nodes"] if node["id"] == "worker:d1_w0_running")

    assert snapshot["run"]["status"] == "finished"
    assert snapshot["counts"]["running"] == 0
    assert worker["status"] == "interrupted"
    assert worker["stale"]
    assert any(file["path"] == str(prompt_path) for file in worker["files"])


def test_monitor_snapshot_shows_resume_node_while_continuing(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("continue old task")
    old_candidate = BeamCandidate(
        id="d0_w0_old",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-old", label="old", instruction="old"),
        score=0.1,
        status="blocked",
        summary="old blocked branch",
        next_context="old",
        prompt_path="",
        response_path="",
    )
    store.write_candidate(run, old_candidate)
    store.write_json(Path(run.root_dir) / "result.json", {"previous": "result"})
    store.append_event(
        run,
        "run.resumed",
        {"start_depth": 1, "resume_message": "", "frontier": []},
    )
    prompt_path = store.write_prompt(run, "workers", "d1_w0_pending", "prompt")
    store.append_event(
        run,
        "worker.started",
        {
            "candidate": "d1_w0_pending",
            "depth": 1,
            "index": 0,
            "parent": None,
            "seed": {"label": "pending", "instruction": "pending"},
            "prompt_path": str(prompt_path),
        },
    )

    snapshot = build_monitor_snapshot(store.runs_dir)
    resume = next(node for node in snapshot["nodes"] if node["type"] == "resume")
    old_worker = next(node for node in snapshot["nodes"] if node["id"] == "worker:d0_w0_old")
    pending_worker = next(node for node in snapshot["nodes"] if node["id"] == "worker:d1_w0_pending")

    assert snapshot["run"]["status"] == "running"
    assert resume["status"] == "running"
    assert resume["body"] == "continued without a new user message"
    assert old_worker["stale"]
    assert pending_worker["status"] == "running"


def test_monitor_latest_uses_run_activity_not_directory_mtime(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    older_active = store.create_run("older but active")
    time.sleep(0.01)
    newer_idle = store.create_run("newer but idle")
    time.sleep(0.01)
    store.append_event(older_active, "run.resumed", {"start_depth": 1, "resume_message": ""})

    snapshot = build_monitor_snapshot(store.runs_dir)

    assert snapshot["run"]["run_id"] == older_active.run_id
    assert snapshot["run"]["run_id"] != newer_idle.run_id


def test_monitor_snapshot_keeps_agent_edges_tree_shaped(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("branch")
    parent = BeamCandidate(
        id="d0_parent",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-0", label="parent", instruction="parent"),
        score=0.8,
        status="promising",
        summary="parent",
        next_context="parent context",
        prompt_path="",
        response_path="",
    )
    child = BeamCandidate(
        id="d1_child",
        parent_id="d0_parent",
        depth=1,
        seed=ThoughtSeed(id="seed-1", label="child", instruction="child"),
        score=0.7,
        status="promising",
        summary="child",
        next_context="child context",
        prompt_path="",
        response_path="",
    )
    store.write_candidate(run, parent)
    store.append_event(
        run,
        "manager.started",
        {"name": "manager_d1_d0_parent", "depth": 1, "parent": "d0_parent"},
    )
    store.append_event(
        run,
        "diversity.started",
        {"name": "diversity_d1_d0_parent", "depth": 1, "parent": "d0_parent"},
    )
    store.append_event(
        run,
        "worker.started",
        {
            "candidate": "d1_child",
            "depth": 1,
            "index": 0,
            "parent": "d0_parent",
            "seed": {"label": "child", "instruction": "child"},
        },
    )
    store.write_candidate(run, child)

    snapshot = build_monitor_snapshot(store.runs_dir)

    assert {"from": "worker:d0_parent", "to": "worker:d1_child"} not in snapshot["edges"]
    assert {"from": "diversity:diversity_d1_d0_parent", "to": "worker:d1_child"} in snapshot["edges"]


def test_monitor_snapshot_highlights_kept_search_paths(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("highlight kept paths")
    parent = BeamCandidate(
        id="d0_parent",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-0", label="parent", instruction="parent"),
        score=0.8,
        status="promising",
        summary="parent",
        next_context="parent context",
        prompt_path="",
        response_path="",
    )
    kept_child = BeamCandidate(
        id="d1_kept",
        parent_id="d0_parent",
        depth=1,
        seed=ThoughtSeed(id="seed-1", label="kept", instruction="kept"),
        score=0.7,
        status="promising",
        summary="kept",
        next_context="kept context",
        prompt_path="",
        response_path="",
    )
    dropped_child = BeamCandidate(
        id="d1_dropped",
        parent_id="d0_parent",
        depth=1,
        seed=ThoughtSeed(id="seed-2", label="dropped", instruction="dropped"),
        score=0.2,
        status="blocked",
        summary="dropped",
        next_context="dropped context",
        prompt_path="",
        response_path="",
    )
    store.write_candidate(run, parent)
    store.write_candidate(run, kept_child)
    store.write_candidate(run, dropped_child)
    store.append_event(
        run,
        "frontier.pruned",
        {"depth": 1, "kept": ["d1_kept"], "dropped": ["d1_dropped"]},
    )

    snapshot = build_monitor_snapshot(store.runs_dir)
    nodes = {node["id"]: node for node in snapshot["nodes"]}
    edges = {(edge["from"], edge["to"]): edge for edge in snapshot["edges"]}

    assert nodes["user"]["path_highlight"]
    assert nodes["worker:d0_parent"]["path_highlight"]
    assert nodes["worker:d1_kept"]["path_highlight"]
    assert not nodes["worker:d1_dropped"].get("path_highlight")
    assert edges[("user", "worker:d0_parent")]["path_highlight"]
    assert edges[("worker:d0_parent", "worker:d1_kept")]["path_highlight"]
    assert not edges[("worker:d0_parent", "worker:d1_dropped")].get("path_highlight")


def test_monitor_snapshot_highlights_latest_resume_frontier_only(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("highlight only the current resume frontier")
    old = BeamCandidate(
        id="d0_old",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-old", label="old", instruction="old"),
        score=0.4,
        status="promising",
        summary="old",
        next_context="old",
        prompt_path="",
        response_path="",
    )
    current = BeamCandidate(
        id="d0_current",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed-current", label="current", instruction="current"),
        score=0.8,
        status="promising",
        summary="current",
        next_context="current",
        prompt_path="",
        response_path="",
    )
    store.write_candidate(run, old)
    store.write_candidate(run, current)
    store.append_event(run, "run.resumed", {"start_depth": 1, "resume_message": "", "frontier": ["d0_old"]})
    store.append_event(run, "run.resumed", {"start_depth": 1, "resume_message": "", "frontier": ["d0_current"]})

    snapshot = build_monitor_snapshot(store.runs_dir)
    nodes = {node["id"]: node for node in snapshot["nodes"]}

    assert "current-frontier" not in nodes["worker:d0_old"].get("path_reasons", [])
    assert "current-frontier" in nodes["worker:d0_current"]["path_reasons"]


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
    assert "treeViewport" in page
    assert "treeSvg" in page
    assert "eventTree" in page
    assert "renderSvgTree" in page
    assert "foreignObject" in page
    assert "svgNodeLabel" in page
    assert "svg-label-title" in page
    assert "path-highlight" in page
    assert "path_highlight" in page
    assert "sortTreeChildren" in page
    assert "nodePriority" in page
    assert "node.stale" in page
    assert "#f8caca" not in page
    assert "renderAgentForest" in page
    assert "objectTree" in page
    assert "renderFileTree" in page
