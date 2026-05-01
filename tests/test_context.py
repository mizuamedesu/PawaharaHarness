from __future__ import annotations

import json
import os
from pathlib import Path

from pawahara_harness.context import BeamCandidate, ContextStore, ThoughtSeed


def test_write_json_skips_unchanged_file_rewrite(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    path = tmp_path / "artifact.json"

    store.write_json(path, {"value": 1})
    fixed_time_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed_time_ns, fixed_time_ns))

    store.write_json(path, {"value": 1})

    assert path.stat().st_mtime_ns == fixed_time_ns
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}

    store.write_json(path, {"value": 2})

    assert path.stat().st_mtime_ns != fixed_time_ns
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}


def test_write_candidate_skips_unchanged_json_but_keeps_events(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("skip unchanged candidate")
    prompt_path = store.write_prompt(run, "workers", "candidate", "prompt")
    response_path = store.write_response(run, "workers", "candidate", "response")
    candidate = BeamCandidate(
        id="candidate",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed", label="label", instruction="instruction"),
        score=0.5,
        status="promising",
        summary="summary",
        next_context="context",
        prompt_path=str(prompt_path),
        response_path=str(response_path),
        artifacts=(),
    )

    candidate_path = store.write_candidate(run, candidate)
    fixed_time_ns = 1_700_000_000_000_000_000
    os.utime(candidate_path, ns=(fixed_time_ns, fixed_time_ns))
    store.write_candidate(run, candidate)

    events = store.list_events(run, limit=10)
    completed = [event for event in events if event["kind"] == "candidate.completed"]
    assert candidate_path.stat().st_mtime_ns == fixed_time_ns
    assert len(completed) == 2
