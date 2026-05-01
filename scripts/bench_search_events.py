from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import pstats
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from pawahara_harness import cli
from pawahara_harness.agents import AgentLaunchSpec, AgentResult
from pawahara_harness.context import (
    BeamCandidate,
    ContextStore,
    RunRecord,
    ThoughtSeed,
    beam_candidate_to_dict,
    run_record_to_dict,
)
from pawahara_harness.web import build_monitor_snapshot


class NoopRuntime:
    def __init__(self, *, duplicate_response: bool = False, empty_response: bool = False, response_bytes: int = 0) -> None:
        self.calls = 0
        self.duplicate_response = duplicate_response
        self.empty_response = empty_response
        self.response_bytes = max(0, response_bytes)

    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        self.calls += 1
        if self.empty_response:
            stdout = ""
        else:
            stdout = self._response_stdout()
        return AgentResult(
            name=spec.name,
            role=spec.role,
            sandbox_id="noop",
            command=spec.command,
            stdout=stdout,
            stderr="",
            exit_code=0,
        )

    def _response_stdout(self) -> str:
        summary = "candidate duplicate" if self.duplicate_response else f"candidate {self.calls}"
        payload = {
            "status": "promising",
            "summary": summary,
            "score": 0.5,
            "novelty": "benchmark",
            "next_context": "benchmark context",
            "artifacts": [],
        }
        if self.response_bytes:
            payload["padding"] = "x" * self.response_bytes
        return json.dumps(payload)


class LegacyContextStore(ContextStore):
    """Pre-optimization write behavior for same-process before/after timings."""

    def role_dir(self, run: RunRecord, role: str) -> Path:
        path = Path(run.root_dir) / role
        path.mkdir(parents=True, exist_ok=True)
        return path

    @contextlib.contextmanager
    def buffered_events(self, run: RunRecord, *, max_lines: int = 64):
        yield

    def write_candidate(self, run: RunRecord, candidate: BeamCandidate) -> Path:
        path = self.role_dir(run, "candidates") / f"{candidate.id}.json"
        payload = beam_candidate_to_dict(candidate)
        self.write_json(path, payload)
        self.append_event(run, "candidate.completed", {"candidate": candidate.id, **payload})
        return path

    def append_event(self, run: RunRecord, kind: str, payload: dict) -> None:
        event = {"ts": "benchmark", "kind": kind, "payload": payload}
        path = Path(run.root_dir) / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_json(self, path: Path, payload: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def write_text(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_response(self, run: RunRecord, role: str, name: str, content: str) -> Path:
        return self.write_text(self.role_dir(run, role) / f"{name}.response.txt", content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark `pawahara-harness search` I/O with a no-op runtime.")
    parser.add_argument("--candidates", type=int, default=528)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--mode", choices=("cli", "persistence", "both"), default="persistence")
    parser.add_argument("--updates", type=int, default=3, help="Candidate JSON rewrites per fake candidate.")
    parser.add_argument("--events-per-candidate", type=int, default=3, help="Extra progress events per fake candidate.")
    parser.add_argument("--duplicate-response", action="store_true")
    parser.add_argument("--empty-response", action="store_true")
    parser.add_argument("--response-bytes", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-lines", type=int, default=20)
    args = parser.parse_args()

    payload = {
        "command": "PYTHONPATH=src python3 scripts/bench_search_events.py",
        "parameters": {
            "candidates": args.candidates,
            "iterations": args.iterations,
            "updates": args.updates,
            "events_per_candidate": args.events_per_candidate,
            "duplicate_response": args.duplicate_response,
            "empty_response": args.empty_response,
            "response_bytes": args.response_bytes,
        },
    }
    if args.mode in {"persistence", "both"}:
        payload["persistence_results"] = run_persistence_benchmark(args)
    if args.mode in {"cli", "both"}:
        cli_results, profiles = run_cli_benchmark(args)
        payload["cli_results"] = cli_results
        if profiles:
            payload["profiles"] = profiles
    print(json.dumps(payload, indent=2))
    return 0


def run_cli_benchmark(args: argparse.Namespace) -> tuple[list[dict], dict[str, str]]:
    results = []
    profiles = {}
    for workers in args.workers:
        times = []
        last_validation = {}
        for index in range(args.iterations):
            with tempfile.TemporaryDirectory() as tmp:
                output = io.StringIO()
                runs_dir = Path(tmp) / "runs"
                argv = [
                    "search",
                    "--goal",
                    "benchmark search event persistence",
                    "--command",
                    "noop",
                    "--runs-dir",
                    str(runs_dir),
                    "--beam-width",
                    str(args.candidates),
                    "--branch-factor",
                    str(args.candidates),
                    "--max-depth",
                    "1",
                    "--max-workers",
                    str(workers),
                    "--no-agentic-roles",
                    "--no-crow",
                    "--no-stop-on-solved",
                ]
                runtime = NoopRuntime(
                    duplicate_response=args.duplicate_response,
                    empty_response=args.empty_response,
                    response_bytes=args.response_bytes,
                )

                def run_once() -> int:
                    with patch("pawahara_harness.cli._build_runtime", return_value=runtime):
                        with contextlib.redirect_stdout(output):
                            return cli.main(argv)

                started = time.perf_counter()
                if args.profile and index == 0:
                    profiler = cProfile.Profile()
                    exit_code = profiler.runcall(run_once)
                    profile_output = io.StringIO()
                    pstats.Stats(profiler, stream=profile_output).sort_stats("cumtime").print_stats(args.profile_lines)
                    profiles[str(workers)] = profile_output.getvalue()
                else:
                    exit_code = run_once()
                elapsed = time.perf_counter() - started
                if exit_code != 0:
                    raise SystemExit(f"benchmark CLI failed with exit code {exit_code}")
                payload = json.loads(output.getvalue())
                run_root = Path(payload["run"]["root_dir"])
                event_kinds = [
                    json.loads(line)["kind"]
                    for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                candidate_count = len(payload["candidates"])
                file_count = sum(1 for path in run_root.rglob("*") if path.is_file())
                expected_file_count = 3 + (3 * candidate_count)
                if candidate_count != args.candidates:
                    raise SystemExit(f"expected {args.candidates} candidates, got {candidate_count}")
                if file_count != expected_file_count:
                    raise SystemExit(f"expected {expected_file_count} files, got {file_count}")
                if event_kinds[0] != "run.created" or event_kinds[-1] != "frontier.pruned":
                    raise SystemExit(f"unexpected event boundaries: {event_kinds[:1]} ... {event_kinds[-1:]}")
                validate_cli_contract(run_root, payload)
                last_validation = {
                    "candidates": candidate_count,
                    "files": file_count,
                    "events": len(event_kinds),
                    "first_event": event_kinds[0],
                    "last_event": event_kinds[-1],
                    "duplicate_response": args.duplicate_response,
                    "empty_response": args.empty_response,
                    "response_bytes": args.response_bytes,
                }
                times.append(elapsed)

        results.append(
            {
                "workers": workers,
                "candidates": args.candidates,
                "iterations": args.iterations,
                "median_seconds": statistics.median(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "validation": last_validation,
            }
        )

    return results, profiles


def run_persistence_benchmark(args: argparse.Namespace) -> list[dict]:
    results = []
    for variant, store_factory in (
        ("legacy", LegacyContextStore),
        ("current", ContextStore),
    ):
        times = []
        last_validation = {}
        for _ in range(args.iterations):
            with tempfile.TemporaryDirectory() as tmp:
                store = store_factory(Path(tmp) / "runs")
                started = time.perf_counter()
                run, candidates = write_fake_candidate_artifacts(
                    store,
                    candidates=args.candidates,
                    updates=args.updates,
                    events_per_candidate=args.events_per_candidate,
                    duplicate_response=args.duplicate_response,
                    empty_response=args.empty_response,
                    response_bytes=args.response_bytes,
                )
                elapsed = time.perf_counter() - started
                times.append(elapsed)
                last_validation = validate_persistence_contract(run, candidates)
        results.append(
            {
                "variant": variant,
                "candidates": args.candidates,
                "iterations": args.iterations,
                "median_seconds": statistics.median(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "validation": last_validation,
            }
        )
    return results


def write_fake_candidate_artifacts(
    store: ContextStore,
    *,
    candidates: int,
    updates: int,
    events_per_candidate: int,
    duplicate_response: bool,
    empty_response: bool,
    response_bytes: int,
) -> tuple[RunRecord, list[BeamCandidate]]:
    run = store.create_run("benchmark search-time persistence", metadata={"benchmark": "persistence"})
    completed: list[BeamCandidate] = []
    with store.buffered_events(run):
        for index in range(candidates):
            candidate_id = f"d0_w{index}_bench"
            prompt_path = store.write_prompt(run, "workers", candidate_id, f"prompt {index}\n" * 4)
            store.append_event(
                run,
                "worker.started",
                {
                    "candidate": candidate_id,
                    "depth": 0,
                    "index": index,
                    "prompt_path": str(prompt_path),
                },
            )
            for event_index in range(events_per_candidate):
                store.append_event(
                    run,
                    "worker.progress",
                    {
                        "candidate": candidate_id,
                        "index": index,
                        "step": event_index,
                        "message": "synthetic progress event",
                    },
                )
            response_path = store.write_response(
                run,
                "workers",
                candidate_id,
                fake_response(
                    index,
                    duplicate_response=duplicate_response,
                    empty_response=empty_response,
                    response_bytes=response_bytes,
                ),
            )
            candidate = None
            for update_index in range(max(1, updates)):
                candidate = BeamCandidate(
                    id=candidate_id,
                    parent_id=None,
                    depth=0,
                    seed=ThoughtSeed(
                        id=f"seed_{index}",
                        label="io-contract-benchmark",
                        instruction="stress persistence writes",
                        novelty_targets=("search-time persistence", "output contract"),
                    ),
                    score=0.1 + (update_index / max(1, updates)),
                    status="promising",
                    summary=f"candidate {index} update {update_index}",
                    next_context=f"context {index} update {update_index}",
                    prompt_path=str(prompt_path),
                    response_path=str(response_path),
                    artifacts=(str(response_path),),
                )
                store.write_candidate(run, candidate)
            assert candidate is not None
            store.append_event(
                run,
                "worker.invocation",
                {
                    "candidate": candidate_id,
                    "exit_code": 0,
                    "sandbox_id": "bench",
                    "command": "noop",
                },
            )
            completed.append(candidate)
        store.append_event(
            run,
            "frontier.pruned",
            {
                "depth": 0,
                "kept": [candidate.id for candidate in completed],
                "dropped": [],
            },
        )
    store.write_json(
        Path(run.root_dir) / "result.json",
        {
            "run": run_record_to_dict(run),
            "best_candidate": beam_candidate_to_dict(completed[-1]) if completed else None,
            "frontier": [beam_candidate_to_dict(candidate) for candidate in completed],
            "candidates": [beam_candidate_to_dict(candidate) for candidate in completed],
            "crow_nudges": [],
        },
    )
    return run, completed


def fake_response(
    index: int,
    *,
    duplicate_response: bool = False,
    empty_response: bool = False,
    response_bytes: int = 0,
) -> str:
    if empty_response:
        return ""
    summary = "candidate duplicate" if duplicate_response else f"candidate {index}"
    payload = {
        "status": "promising",
        "summary": summary,
        "score": 0.5,
        "novelty": "benchmark",
        "next_context": "benchmark context",
        "artifacts": [],
    }
    if response_bytes:
        payload["padding"] = "x" * response_bytes
    return json.dumps(
        payload
    )


def validate_persistence_contract(run: RunRecord, candidates: list[BeamCandidate]) -> dict:
    run_root = Path(run.root_dir)
    events = [json.loads(line) for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_files = expected_run_files(run_root, candidates)
    actual_files = {path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)[:5]
        extra = sorted(actual_files - expected_files)[:5]
        raise SystemExit(f"file contract failed: missing={missing} extra={extra}")
    if not events or events[0]["kind"] != "run.created" or events[-1]["kind"] != "frontier.pruned":
        raise SystemExit("event boundary contract failed")
    validate_event_objects(events)
    validate_candidate_event_order(events, candidates)
    validate_candidate_artifacts(run_root, candidates)
    validate_result_shape(run_root, candidates)
    snapshot = build_monitor_snapshot(run_root.parent, run_id=run.run_id)
    if snapshot["counts"]["candidates"] != len(candidates):
        raise SystemExit(f"snapshot expected {len(candidates)} candidates, got {snapshot['counts']['candidates']}")
    first = candidates[0] if candidates else None
    if first:
        candidate_json = run_root / "candidates" / f"{first.id}.json"
        candidate_payload = json.loads(candidate_json.read_text(encoding="utf-8"))
        if candidate_payload["summary"] != first.summary:
            raise SystemExit("candidate JSON did not preserve latest update")
        worker = next(node for node in snapshot["nodes"] if node["id"] == f"worker:{first.id}")
        labels = {file["label"] for file in worker["files"]}
        if not {"prompt", "response", "candidate.json"}.issubset(labels):
            raise SystemExit(f"worker files missing UI-visible labels: {labels}")
    return {
        "files": len(actual_files),
        "events": len(events),
        "snapshot_candidates": snapshot["counts"]["candidates"],
        "first_event": events[0]["kind"] if events else None,
        "last_event": events[-1]["kind"] if events else None,
        "candidate_json_keys": sorted(beam_candidate_to_dict(candidates[0]).keys()) if candidates else [],
        "event_keys": sorted(events[0].keys()) if events else [],
    }


def validate_cli_contract(run_root: Path, payload: dict) -> None:
    candidates = [candidate for candidate in payload.get("candidates", []) if isinstance(candidate, dict)]
    events = [json.loads(line) for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_files = {"run.json", "events.jsonl", "result.json"}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        expected_files.add(f"workers/{candidate_id}.prompt.md")
        expected_files.add(f"workers/{candidate_id}.response.txt")
        expected_files.add(f"candidates/{candidate_id}.json")
    actual_files = {path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise SystemExit("CLI file contract failed")
    validate_event_objects(events)
    validate_candidate_payloads(run_root, candidates)
    validate_candidate_event_order_for_ids(events, [str(candidate["id"]) for candidate in candidates])


def expected_run_files(run_root: Path, candidates: list[BeamCandidate]) -> set[str]:
    files = {"run.json", "events.jsonl", "result.json"}
    for candidate in candidates:
        files.add(Path(candidate.prompt_path).relative_to(run_root).as_posix())
        files.add(Path(candidate.response_path).relative_to(run_root).as_posix())
        files.add(f"candidates/{candidate.id}.json")
    return files


def validate_event_objects(events: list[dict]) -> None:
    for index, event in enumerate(events):
        if set(event) != {"ts", "kind", "payload"}:
            raise SystemExit(f"event {index} keys changed: {sorted(event)}")
        if not isinstance(event["payload"], dict):
            raise SystemExit(f"event {index} payload is not an object")


def validate_candidate_event_order(events: list[dict], candidates: list[BeamCandidate]) -> None:
    validate_candidate_event_order_for_ids(events, [candidate.id for candidate in candidates])


def validate_candidate_event_order_for_ids(events: list[dict], candidate_ids: list[str]) -> None:
    positions: dict[tuple[str, str], int] = {}
    completed_counts = {candidate_id: 0 for candidate_id in candidate_ids}
    for index, event in enumerate(events):
        payload = event.get("payload", {})
        candidate_id = payload.get("candidate") if isinstance(payload, dict) else None
        if not candidate_id:
            continue
        kind = str(event.get("kind"))
        positions.setdefault((str(candidate_id), kind), index)
        if kind == "candidate.completed" and str(candidate_id) in completed_counts:
            completed_counts[str(candidate_id)] += 1
    for candidate_id in candidate_ids:
        try:
            started = positions[(candidate_id, "worker.started")]
            completed = positions[(candidate_id, "candidate.completed")]
            invocation = positions[(candidate_id, "worker.invocation")]
        except KeyError as exc:
            raise SystemExit(f"missing event for candidate {candidate_id}: {exc}") from exc
        if not started < completed < invocation:
            raise SystemExit(f"event order changed for candidate {candidate_id}")
        if completed_counts[candidate_id] < 1:
            raise SystemExit(f"candidate.completed missing for {candidate_id}")


def validate_candidate_artifacts(run_root: Path, candidates: list[BeamCandidate]) -> None:
    validate_candidate_payloads(run_root, [beam_candidate_to_dict(candidate) for candidate in candidates])


def validate_candidate_payloads(run_root: Path, candidates: list[dict]) -> None:
    expected_keys = set(beam_candidate_to_dict(sample_candidate()).keys())
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        candidate_json = run_root / "candidates" / f"{candidate_id}.json"
        payload = json.loads(candidate_json.read_text(encoding="utf-8"))
        if set(payload) != expected_keys:
            raise SystemExit(f"candidate JSON keys changed for {candidate_id}: {sorted(payload)}")
        for path_key in ("prompt_path", "response_path"):
            path = Path(payload[path_key])
            try:
                path.read_text(encoding="utf-8")
            except OSError as exc:
                raise SystemExit(f"{path_key} is unreadable for {candidate_id}") from exc
        for artifact in payload.get("artifacts", []) or []:
            artifact_path = Path(artifact)
            if not artifact_path.exists():
                raise SystemExit(f"artifact is unreadable for {candidate_id}: {artifact}")


def validate_result_shape(run_root: Path, candidates: list[BeamCandidate]) -> None:
    result = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
    expected_result_keys = {"run", "best_candidate", "frontier", "candidates", "crow_nudges"}
    if set(result) != expected_result_keys:
        raise SystemExit(f"result JSON keys changed: {sorted(result)}")
    if len(result["candidates"]) != len(candidates):
        raise SystemExit("result candidate count changed")


def sample_candidate() -> BeamCandidate:
    return BeamCandidate(
        id="sample",
        parent_id=None,
        depth=0,
        seed=ThoughtSeed(id="seed", label="label", instruction="instruction"),
        score=0.0,
        status="promising",
        summary="summary",
        next_context="context",
        prompt_path="prompt",
        response_path="response",
        artifacts=(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
