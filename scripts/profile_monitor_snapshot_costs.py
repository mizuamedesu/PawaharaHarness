from __future__ import annotations

import argparse
import builtins
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench_monitor_snapshot import make_run
from pawahara_harness import web


class CostMap:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, bucket: str, elapsed: float) -> None:
        self.totals[bucket] = self.totals.get(bucket, 0.0) + elapsed
        self.counts[bucket] = self.counts.get(bucket, 0) + 1

    def wrap(self, bucket: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.add(bucket, time.perf_counter() - started)

        return wrapped

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"bucket": bucket, "seconds": self.totals[bucket], "calls": self.counts[bucket]}
            for bucket in sorted(self.totals, key=self.totals.get, reverse=True)
        ]


def profile_snapshot(
    runs_dir: Path,
    *,
    iterations: int,
    run_id: str = "run_bench",
) -> dict[str, Any]:
    costs = CostMap()
    original: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, replacement: Any) -> None:
        original.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    patch(web, "build_nodes_and_edges", costs.wrap("build_nodes_and_edges", web.build_nodes_and_edges))
    patch(
        web,
        "read_candidate_files_with_preview_models",
        costs.wrap("read_candidate_files_with_preview_models", web.read_candidate_files_with_preview_models),
    )
    patch(web, "list_run_files", costs.wrap("list_run_files", web.list_run_files))
    patch(web, "files_from_payload", costs.wrap("files_from_payload", web.files_from_payload))
    patch(web, "files_from_candidate", costs.wrap("files_from_candidate", web.files_from_candidate))
    patch(web, "file_item", costs.wrap("file_item", web.file_item))
    patch(web, "read_text_preview", costs.wrap("read_text_preview", web.read_text_preview))
    patch(web, "dedupe_files", costs.wrap("dedupe_files", web.dedupe_files))
    patch(web, "short_text", costs.wrap("render_formatting.short_text", web.short_text))
    patch(web, "worker_title", costs.wrap("render_formatting.worker_title", web.worker_title))
    patch(web, "manager_title", costs.wrap("render_formatting.manager_title", web.manager_title))
    patch(web, "diversity_title", costs.wrap("render_formatting.diversity_title", web.diversity_title))
    patch(web.os, "stat", costs.wrap("path_io.os_stat", web.os.stat))
    patch(web.os, "scandir", costs.wrap("path_io.os_scandir", web.os.scandir))
    patch(web.os, "fspath", costs.wrap("path_format.os_fspath", web.os.fspath))
    patch(web.os.path, "join", costs.wrap("path_format.os_path_join", web.os.path.join))
    patch(web.os.path, "basename", costs.wrap("path_format.os_path_basename", web.os.path.basename))
    patch(builtins, "open", costs.wrap("path_io.open", builtins.open))

    times: list[float] = []
    snapshot: dict[str, Any] = {}
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            snapshot = web.build_monitor_snapshot(runs_dir, run_id=run_id)
            elapsed = time.perf_counter() - started
            costs.add("build_monitor_snapshot.total", elapsed)
            times.append(elapsed)
    finally:
        for owner, name, value in reversed(original):
            setattr(owner, name, value)

    warm_times = times[1:] if len(times) > 1 else times
    return {
        "iterations": iterations,
        "times": times,
        "cold_seconds": times[0] if times else None,
        "warm_median_seconds": statistics.median(warm_times) if warm_times else None,
        "nodes": len(snapshot.get("nodes", [])),
        "events": len(snapshot.get("events", [])),
        "files": len(snapshot.get("files", [])),
        "costs": costs.rows(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile monitor snapshot cost buckets on a synthetic run.")
    parser.add_argument("--candidates", type=int, default=3000)
    parser.add_argument("--response-bytes", type=int, default=65536)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = make_run(Path(tmp), candidates=args.candidates, response_bytes=args.response_bytes)
        result = profile_snapshot(runs_dir, iterations=args.iterations)
        result.update(
            {
                "command": "python3 scripts/profile_monitor_snapshot_costs.py",
                "candidates": args.candidates,
                "response_bytes": args.response_bytes,
            }
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
