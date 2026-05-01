from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from pawahara_harness.web import build_monitor_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark monitor snapshot generation on a synthetic run.")
    parser.add_argument("--candidates", type=int, default=1500)
    parser.add_argument("--response-bytes", type=int, default=65536)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = make_run(Path(tmp), candidates=args.candidates, response_bytes=args.response_bytes)
        times = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            snapshot = build_monitor_snapshot(runs_dir, run_id="run_bench")
            times.append(time.perf_counter() - started)

        print(
            json.dumps(
                {
                    "command": "python3 scripts/bench_monitor_snapshot.py",
                    "candidates": args.candidates,
                    "response_bytes": args.response_bytes,
                    "iterations": args.iterations,
                    "median_seconds": statistics.median(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "nodes": len(snapshot["nodes"]),
                    "events": len(snapshot["events"]),
                    "files": len(snapshot["files"]),
                },
                indent=2,
            )
        )
    return 0


def make_run(root: Path, *, candidates: int, response_bytes: int) -> Path:
    runs_dir = root / "runs"
    run_dir = runs_dir / "run_bench"
    workers_dir = run_dir / "workers"
    candidates_dir = run_dir / "candidates"
    workers_dir.mkdir(parents=True)
    candidates_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run_bench",
                "goal": "benchmark monitor snapshot generation",
                "created_at": "benchmark",
                "root_dir": str(run_dir),
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    response_text = "x" * response_bytes
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as events:
        for index in range(candidates):
            candidate_id = f"d0_w{index}_bench"
            prompt_path = workers_dir / f"{candidate_id}.prompt.md"
            response_path = workers_dir / f"{candidate_id}.response.txt"
            prompt_path.write_text(f"prompt {index}", encoding="utf-8")
            response_path.write_text(response_text, encoding="utf-8")
            seed = {
                "id": "seed_bench",
                "label": "benchmark",
                "instruction": "exercise monitor file previews",
                "novelty_targets": ["regression prevention"],
            }
            candidate = {
                "id": candidate_id,
                "parent_id": None,
                "depth": 0,
                "seed": seed,
                "score": 0.5,
                "status": "promising",
                "summary": "benchmark candidate",
                "next_context": "benchmark context",
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "artifacts": [],
            }
            (candidates_dir / f"{candidate_id}.json").write_text(json.dumps(candidate), encoding="utf-8")
            events.write(
                json.dumps(
                    {
                        "ts": "benchmark",
                        "kind": "worker.started",
                        "payload": {
                            "candidate": candidate_id,
                            "depth": 0,
                            "index": index,
                            "parent": None,
                            "seed": seed,
                            "prompt_path": str(prompt_path),
                        },
                    }
                )
                + "\n"
            )
            events.write(
                json.dumps(
                    {
                        "ts": "benchmark",
                        "kind": "worker.invocation",
                        "payload": {
                            "candidate": candidate_id,
                            "exit_code": 0,
                            "sandbox_id": "local",
                            "command": "agent",
                        },
                    }
                )
                + "\n"
            )
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    return runs_dir


if __name__ == "__main__":
    raise SystemExit(main())
