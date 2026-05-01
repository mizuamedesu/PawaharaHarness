from __future__ import annotations

import json
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_MONITOR_HOST = "127.0.0.1"
DEFAULT_MONITOR_PORT = 8765
MAX_TEXT_PREVIEW_CHARS = 4000
MAX_TEXT_PREVIEW_CACHE_ITEMS = 10000

_TEXT_PREVIEW_CACHE: dict[str, tuple[int, int, int, str]] = {}
_TEXT_PREVIEW_CACHE_LOCK = threading.Lock()
_CANDIDATE_FILE_CACHE: dict[str, tuple[int, int, int, dict[str, Any], str]] = {}
_CANDIDATE_FILE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class PayloadFileRef:
    path: str
    label: str


@dataclass(frozen=True, slots=True)
class CandidateFileModel:
    candidate_item: dict[str, Any]
    payload_refs: tuple[PayloadFileRef, ...]


@dataclass
class AgentMonitorServer:
    runs_dir: Path
    host: str = DEFAULT_MONITOR_HOST
    port: int = DEFAULT_MONITOR_PORT
    httpd: ThreadingHTTPServer | None = field(default=None, init=False)
    thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> AgentMonitorServer:
        if self.httpd is not None:
            return self

        handler = make_monitor_handler(Path(self.runs_dir))
        ports = [0] if self.port == 0 else list(range(self.port, self.port + 50))
        last_error: OSError | None = None
        for port in ports:
            try:
                httpd = ReusableThreadingHTTPServer((self.host, port), handler)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"could not start agent monitor: {last_error}")

        self.httpd = httpd
        self.port = int(httpd.server_address[1])
        self.thread = threading.Thread(target=httpd.serve_forever, name="pawahara-agent-monitor", daemon=True)
        self.thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        if self.httpd is None:
            return
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.httpd = None
        self.thread = None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def make_monitor_handler(runs_dir: Path) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"", "/"}:
                self._send_html(render_monitor_page())
                return
            if parsed.path == "/api/latest":
                query = parse_qs(parsed.query)
                run_id = query.get("run", [None])[0]
                self._send_json(build_monitor_snapshot(runs_dir, run_id=run_id))
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                path_text = query.get("path", [""])[0]
                payload, status = read_monitor_file(runs_dir, path_text)
                self._send_json(payload, status=status)
                return
            if parsed.path.startswith("/api/runs/"):
                run_id = parsed.path.removeprefix("/api/runs/").strip("/")
                self._send_json(build_monitor_snapshot(runs_dir, run_id=run_id or None))
                return
            self.send_error(404)

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body_text: str, status: int = 200) -> None:
            body = body_text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MonitorHandler


def build_monitor_snapshot(runs_dir: Path, *, run_id: str | None = None) -> dict[str, Any]:
    run_dir = select_run_dir(Path(runs_dir), run_id=run_id)
    if run_dir is None:
        return {
            "ok": True,
            "run": None,
            "nodes": [],
            "edges": [],
            "events": [],
            "files": [],
            "role_states": [],
            "result": None,
            "counts": {"candidates": 0, "running": 0, "completed": 0},
        }

    run_data = read_json(run_dir / "run.json") or {}
    events = read_events(run_dir / "events.jsonl")
    candidates, candidate_file_previews, candidate_file_models = read_candidate_files_with_preview_models(run_dir / "candidates")
    result_data = read_json(run_dir / "result.json")
    role_states = read_role_states(run_dir / "roles")
    result_exists = (run_dir / "result.json").exists()
    nodes, edges = build_nodes_and_edges(
        run_dir,
        run_data,
        events,
        candidates,
        candidate_file_previews=candidate_file_previews,
        candidate_file_models=candidate_file_models,
        result_exists=result_exists,
    )
    mark_interrupted_workers(nodes, events, result_exists=result_exists)
    running = sum(
        1
        for node in nodes
        if node.get("status") == "running" and node.get("type") not in {"user", "resume"}
    )
    completed = sum(1 for node in nodes if node.get("type") == "worker" and node.get("status") != "running")
    run_status = "running" if running else "finished" if result_exists else "interrupted"
    latest_running_depth = max((node_depth(node) for node in nodes if node.get("type") == "worker" and node.get("status") == "running"), default=None)
    has_resume = any(str(event.get("kind", "")) == "run.resumed" for event in events)
    latest_resume_id = latest_resume_node_id(events)
    for node in nodes:
        if node.get("id") == "user":
            node["status"] = run_status
        if node.get("type") == "resume":
            if run_status == "running" and node.get("id") == latest_resume_id:
                node["status"] = "running"
            elif run_status == "interrupted" and node.get("id") == latest_resume_id:
                node["status"] = "interrupted"
                node["stale"] = True
            else:
                node["status"] = "done"
        if (
            has_resume
            and run_status == "running"
            and latest_running_depth is not None
            and node.get("type") == "worker"
            and node.get("status") in {"blocked", "dead_end"}
            and node_depth(node) < latest_running_depth
        ):
            node["stale"] = True
    apply_search_path_highlights(nodes, edges, events, candidates, result_data)
    return {
        "ok": True,
        "run": {
            "run_id": run_data.get("run_id", run_dir.name),
            "goal": run_data.get("goal", ""),
            "created_at": run_data.get("created_at", ""),
            "root_dir": str(run_dir),
            "metadata": run_data.get("metadata", {}),
            "status": run_status,
        },
        "nodes": nodes,
        "edges": edges,
        "events": events,
        "files": list_run_files(run_dir),
        "role_states": role_states,
        "result": result_data,
        "counts": {"candidates": len(candidates), "running": running, "completed": completed},
    }


def select_run_dir(runs_dir: Path, *, run_id: str | None = None) -> Path | None:
    if run_id:
        direct = Path(run_id)
        candidates = [direct, runs_dir / run_id]
        for candidate in candidates:
            if (candidate / "run.json").exists():
                return candidate
        return None
    if not runs_dir.exists():
        return None
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir() and (path / "run.json").exists()]
    if not run_dirs:
        return None
    return max(run_dirs, key=run_activity_mtime)


def run_activity_mtime(run_dir: Path) -> float:
    paths = [
        run_dir,
        run_dir / "run.json",
        run_dir / "events.jsonl",
        run_dir / "result.json",
    ]
    for dirname in ("workers", "manager", "diversity", "crow", "candidates", "roles"):
        directory = run_dir / dirname
        if directory.exists():
            paths.extend(path for path in directory.glob("*") if path.is_file())
    latest = 0.0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def read_candidate_files(candidate_dir: Path) -> list[dict[str, Any]]:
    candidates, _previews = read_candidate_files_with_previews(candidate_dir)
    return candidates


def read_candidate_files_with_previews(candidate_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates, previews, _file_models = read_candidate_files_with_preview_models(candidate_dir)
    return candidates, previews


def read_candidate_files_with_preview_models(
    candidate_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, CandidateFileModel]]:
    if not candidate_dir.exists():
        return [], {}, {}
    candidates: list[dict[str, Any]] = []
    previews: dict[str, str] = {}
    file_models: dict[str, CandidateFileModel] = {}
    candidate_dir_text = os.fspath(candidate_dir)
    try:
        with os.scandir(candidate_dir_text) as scanner:
            entries = []
            for entry in scanner:
                if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    path_stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                entries.append((entry.name, entry.path, path_stat))
            entries.sort(key=lambda item: item[0])
    except OSError:
        return [], {}, {}
    for _name, path_text, path_stat in entries:
        cache_signature = (path_stat.st_mtime_ns, path_stat.st_ctime_ns, path_stat.st_size)
        with _CANDIDATE_FILE_CACHE_LOCK:
            cached = _CANDIDATE_FILE_CACHE.get(path_text)
            if cached is not None and cached[:3] == cache_signature:
                data = dict(cached[3])
                candidates.append(data)
                previews[path_text] = cached[4]
                model = candidate_file_model_from_stat(path_text, path_stat, cached[4], data)
                file_models[path_text] = model
                continue
        try:
            with open(path_text, encoding="utf-8") as handle:
                text = handle.read()
            data = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            candidates.append(data)
            preview = text[:MAX_TEXT_PREVIEW_CHARS]
            previews[path_text] = preview
            model = candidate_file_model_from_stat(path_text, path_stat, preview, data)
            file_models[path_text] = model
            with _CANDIDATE_FILE_CACHE_LOCK:
                if len(_CANDIDATE_FILE_CACHE) >= MAX_TEXT_PREVIEW_CACHE_ITEMS:
                    _CANDIDATE_FILE_CACHE.clear()
                _CANDIDATE_FILE_CACHE[path_text] = (*cache_signature, dict(data), preview)
    return candidates, previews, file_models


def candidate_file_model_from_stat(
    path_text: str,
    path_stat: os.stat_result,
    preview: str,
    candidate: dict[str, Any],
) -> CandidateFileModel:
    return CandidateFileModel(
        candidate_item={
            "label": "candidate.json",
            "path": path_text,
            "size": path_stat.st_size,
            "preview": preview,
        },
        payload_refs=payload_file_refs(candidate, include_artifacts=True),
    )


def read_role_states(role_dir: Path) -> list[dict[str, Any]]:
    if not role_dir.exists():
        return []
    states: list[dict[str, Any]] = []
    for path in sorted(role_dir.glob("*.json")):
        data = read_json(path)
        if data:
            states.append({"path": str(path), "data": data})
    return states


def list_run_files(run_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not run_dir.exists():
        return files
    run_dir_text = os.fspath(run_dir)

    def collect(dirpath: str, relative_dir: str = "") -> None:
        try:
            entries = list(os.scandir(dirpath))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                relative_path = os.path.join(relative_dir, entry.name) if relative_dir else entry.name
                collect(entry.path, relative_path)
                continue
            try:
                path_stat = entry.stat()
            except OSError:
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                continue
            relative_path = os.path.join(relative_dir, entry.name) if relative_dir else entry.name
            files.append(
                {
                    "path": entry.path,
                    "relative_path": relative_path,
                    "size": path_stat.st_size,
                    "mtime": path_stat.st_mtime,
                }
            )

    collect(run_dir_text)
    files.sort(key=lambda item: item["path"])
    return files


def read_monitor_file(runs_dir: Path, path_text: str) -> tuple[dict[str, Any], int]:
    path_text = unquote(path_text).strip()
    if not path_text:
        return {"ok": False, "error": "missing path"}, 400
    path = resolve_monitor_path(Path(runs_dir), path_text)
    if path is None:
        return {"ok": False, "error": "file is outside the monitor runs directory or does not exist"}, 404
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}, 500
    text = raw.decode("utf-8", errors="replace")
    return {
        "ok": True,
        "path": str(path),
        "size": len(raw),
        "content": text,
    }, 200


def resolve_monitor_path(runs_dir: Path, path_text: str) -> Path | None:
    raw = Path(path_text)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, runs_dir / raw]
    try:
        allowed_root = runs_dir.resolve(strict=False)
    except OSError:
        return None
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(allowed_root)
        except ValueError:
            continue
        return resolved
    return None


def build_nodes_and_edges(
    run_dir: Path,
    run_data: dict[str, Any],
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    candidate_file_previews: dict[str, str] | None = None,
    candidate_file_models: dict[str, CandidateFileModel] | None = None,
    result_exists: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    node_order_seen: set[str] = set()
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str]] = set()
    edge_targets_by_source: dict[str, set[str]] = {}
    renderable_incoming_targets: set[str] = set()
    file_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    candidate_dir_text = os.fspath(run_dir / "candidates")

    def upsert(node_id: str, **fields: Any) -> dict[str, Any]:
        node = nodes.setdefault(node_id, {"id": node_id})
        if node_id not in node_order_seen:
            node_order.append(node_id)
            node_order_seen.add(node_id)
        renderable_incoming_targets.update(edge_targets_by_source.get(node_id, ()))
        for key, value in fields.items():
            if value is not None:
                node[key] = value
        return node

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edge_targets_by_source.setdefault(source, set()).add(target)
        if source in nodes:
            renderable_incoming_targets.add(target)
        edges.append({"from": source, "to": target})

    upsert(
        "user",
        type="user",
        title="You",
        status="finished" if result_exists else "running",
        body=short_text(str(run_data.get("goal", "")), 2000),
        meta={"created_at": run_data.get("created_at", "")},
        details={"run": run_data},
        files=existing_files(run_dir / "run.json", run_dir / "events.jsonl", run_dir / "result.json", cache=file_cache),
    )

    for event_index, event in enumerate(events):
        kind = str(event.get("kind", ""))
        payload = event_payload(event)
        if kind == "run.resumed":
            key = resume_node_id(event_index)
            message = str(payload.get("resume_message", "")).strip()
            upsert(
                key,
                type="resume",
                title=f"Resume d{payload.get('start_depth', '?')}",
                status="running",
                body=short_text(message or "continued without a new user message", 1800),
                meta=payload,
                details={"event": event},
                files=[],
            )
            add_edge("user", key)
        elif kind == "manager.started":
            key = manager_node_id(payload)
            parent = payload.get("parent")
            upsert(
                key,
                type="manager",
                title=manager_title(payload),
                status="running",
                body="thinking",
                meta=payload,
                details={"event": event},
                files=files_from_payload(payload, cache=file_cache),
            )
            add_edge(worker_node_id(parent) if parent else "user", key)
        elif kind == "manager.decision":
            key = manager_node_id(payload)
            decision = dict(payload.get("decision") or {})
            upsert(
                key,
                type="manager",
                title=manager_title(payload),
                status="done",
                body=short_text(role_body(decision, ("directive", "rationale", "context_to_keep")), 2400),
                meta=payload,
                details={"event": event, "decision": decision},
                files=files_from_payload(payload, cache=file_cache),
            )
            parent = payload.get("parent")
            add_edge(worker_node_id(parent) if parent else "user", key)
        elif kind == "manager.stop":
            key = manager_node_id(payload)
            upsert(
                key,
                type="manager",
                title=manager_title(payload),
                status="stopped",
                body=short_text(str(payload.get("rationale", "")), 1600),
                meta=payload,
                details={"event": event},
                files=files_from_payload(payload, cache=file_cache),
            )
            parent = payload.get("parent")
            add_edge(worker_node_id(parent) if parent else "user", key)
        elif kind == "diversity.started":
            key = diversity_node_id(payload)
            upsert(
                key,
                type="diversity",
                title=diversity_title(payload),
                status="running",
                body="planning",
                meta=payload,
                details={"event": event},
                files=files_from_payload(payload, cache=file_cache),
            )
            add_edge(manager_node_id(payload), key)
        elif kind == "diversity.plan":
            key = diversity_node_id(payload)
            plan = dict(payload.get("plan") or {})
            labels = ", ".join(seed_label(seed) for seed in plan.get("seeds", []) if seed_label(seed))
            body = "\n".join(part for part in [str(plan.get("rationale", "")).strip(), labels] if part)
            upsert(
                key,
                type="diversity",
                title=diversity_title(payload),
                status="done",
                body=short_text(body, 2400),
                meta=payload,
                details={"event": event, "plan": plan},
                files=files_from_payload(payload, cache=file_cache),
            )
            add_edge(manager_node_id(payload), key)
        elif kind == "worker.started":
            candidate_id = str(payload.get("candidate", ""))
            if not candidate_id:
                continue
            key = worker_node_id(candidate_id)
            seed = payload.get("seed") or {}
            upsert(
                key,
                type="worker",
                title=worker_title(candidate_id, seed_label(seed)),
                status="running",
                body=short_text(str(seed.get("instruction", "")) if isinstance(seed, dict) else "", 1800),
                meta=payload,
                details={"event": event, "seed": seed},
                files=files_from_payload(payload, cache=file_cache),
            )
            add_edge(diversity_node_id(payload), key)
        elif kind == "worker.invocation":
            candidate_id = str(payload.get("candidate", ""))
            if candidate_id:
                existing = nodes.get(worker_node_id(candidate_id), {})
                meta = dict(existing.get("meta") or {})
                meta.update(payload)
                details = dict(existing.get("details") or {})
                details["invocation"] = payload
                upsert(worker_node_id(candidate_id), meta=meta, details=details)
        elif kind == "crow.started":
            key = crow_node_id(payload)
            upsert(
                key,
                type="crow",
                title=crow_title(payload),
                status="running",
                body="checking completion",
                meta=payload,
                details={"event": event},
                files=files_from_payload(payload, cache=file_cache),
            )
            add_edge("user", key)
        elif kind in {"crow.nudge", "crow.verdict"}:
            key = crow_node_id(payload)
            verdict = dict(payload.get("verdict") or payload)
            upsert(
                key,
                type="crow",
                title=crow_title(payload),
                status="done",
                body=short_text(role_body(verdict, ("message", "reason")), 1800),
                meta=payload,
                details={"event": event, "verdict": verdict},
                files=files_from_payload(payload, cache=file_cache),
            )
            add_edge("user", key)

    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if not candidate_id:
            continue
        seed = candidate.get("seed") or {}
        key = worker_node_id(candidate_id)
        existing = nodes.get(key, {})
        details = dict(existing.get("details") or {})
        details["candidate"] = candidate
        file_items = list(existing.get("files") or [])
        file_items.extend(
            files_from_candidate(
                candidate_dir_text,
                candidate,
                candidate_file_previews=candidate_file_previews,
                candidate_file_models=candidate_file_models,
                cache=file_cache,
            )
        )
        upsert(
            key,
            type="worker",
            title=worker_title(candidate_id, seed_label(seed)),
            status=candidate.get("status", "done"),
            score=candidate.get("score"),
            body=short_text(str(candidate.get("summary", "")), 2400),
            meta=candidate,
            details=details,
            files=dedupe_files(file_items),
        )
        parent = candidate.get("parent_id")
        if has_renderable_incoming_edge(renderable_incoming_targets, key):
            continue
        if parent:
            add_edge(worker_node_id(parent), key)
        else:
            diversity_id = diversity_node_id(candidate)
            add_edge(diversity_id if diversity_id in nodes else "user", key)

    add_prompt_only_worker_nodes(
        run_dir,
        nodes,
        node_order,
        edges,
        edge_keys,
        edge_targets_by_source,
        renderable_incoming_targets,
        file_cache,
    )

    return [nodes[node_id] for node_id in node_order], edges


def apply_search_path_highlights(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    result_data: dict[str, Any] | None,
) -> None:
    node_by_id = {str(node.get("id", "")): node for node in nodes}
    candidate_by_id = {str(candidate.get("id", "")): candidate for candidate in candidates if candidate.get("id")}
    incoming: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for edge in edges:
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if not source or not target:
            continue
        incoming.setdefault(target, []).append((source, edge))

    targets: list[tuple[str, str]] = []

    def add_candidate_target(value: Any, reason: str) -> None:
        for candidate_id in candidate_ids_from_value(value):
            if candidate_id in candidate_by_id and candidate_is_viable(candidate_by_id[candidate_id]):
                targets.append((worker_node_id(candidate_id), reason))

    if result_data:
        add_candidate_target(result_data.get("frontier"), "final-frontier")
        add_candidate_target(result_data.get("best_candidate"), "best-path")

    latest_frontier_event = latest_search_frontier_event(events)
    if latest_frontier_event is not None:
        kind = str(latest_frontier_event.get("kind", ""))
        payload = event_payload(latest_frontier_event)
        if kind == "frontier.pruned":
            add_candidate_target(payload.get("kept"), "current-frontier")
        elif kind == "run.resumed":
            add_candidate_target(payload.get("frontier"), "current-frontier")

    if not result_data:
        best_candidate = best_viable_candidate(candidates)
        if best_candidate is not None:
            add_candidate_target(best_candidate, "best-path")

    for node in nodes:
        if node.get("type") == "worker" and node.get("status") == "running":
            targets.append((str(node.get("id", "")), "active-path"))

    highlighted_nodes: set[tuple[str, str]] = set()
    highlighted_edges: set[tuple[str, str, str]] = set()

    def mark_node(node_id: str, reason: str) -> None:
        node = node_by_id.get(node_id)
        if node is None:
            return
        node["path_highlight"] = True
        reasons = node.setdefault("path_reasons", [])
        if isinstance(reasons, list) and reason not in reasons:
            reasons.append(reason)

    def mark_edge(edge: dict[str, Any], reason: str) -> None:
        edge["path_highlight"] = True
        reasons = edge.setdefault("path_reasons", [])
        if isinstance(reasons, list) and reason not in reasons:
            reasons.append(reason)

    def highlight_to_root(node_id: str, reason: str) -> None:
        stack = [node_id]
        while stack:
            current = stack.pop()
            marker = (current, reason)
            if marker in highlighted_nodes:
                continue
            highlighted_nodes.add(marker)
            mark_node(current, reason)
            for source, edge in incoming.get(current, []):
                edge_marker = (source, current, reason)
                if edge_marker not in highlighted_edges:
                    highlighted_edges.add(edge_marker)
                    mark_edge(edge, reason)
                stack.append(source)

    for node_id, reason in targets:
        if node_id:
            highlight_to_root(node_id, reason)


def latest_search_frontier_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if str(event.get("kind", "")) in {"frontier.pruned", "run.resumed"}:
            return event
    return None


def latest_resume_node_id(events: list[dict[str, Any]]) -> str | None:
    latest_index = None
    for index, event in enumerate(events):
        if str(event.get("kind", "")) == "run.resumed":
            latest_index = index
    return resume_node_id(latest_index) if latest_index is not None else None


def best_viable_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [candidate for candidate in candidates if candidate_is_viable(candidate)]
    if not viable:
        return None
    return max(viable, key=lambda candidate: (float(candidate.get("score") or 0.0), int(candidate.get("depth") or 0)))


def candidate_is_viable(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("status", "")) in {"solved", "promising"}


def mark_interrupted_workers(
    nodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    result_exists: bool,
) -> None:
    last_resume_index = max(
        (index for index, event in enumerate(events) if str(event.get("kind", "")) == "run.resumed"),
        default=None,
    )
    started_at: dict[str, int] = {}
    for index, event in enumerate(events):
        if str(event.get("kind", "")) != "worker.started":
            continue
        candidate_id = str(event_payload(event).get("candidate", ""))
        if candidate_id:
            started_at[candidate_id] = index

    for node in nodes:
        if node.get("type") != "worker" or node.get("status") != "running":
            continue
        candidate_id = str(node.get("id", "")).removeprefix("worker:")
        meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
        started_index = started_at.get(candidate_id)
        prompt_only = meta.get("source") == "worker prompt file"
        stale_before_resume = (
            last_resume_index is not None
            and started_index is not None
            and started_index < last_resume_index
        )
        completed_before_orphan = result_exists and last_resume_index is None
        if prompt_only or stale_before_resume or completed_before_orphan:
            node["status"] = "interrupted"
            node["stale"] = True
            details = node.setdefault("details", {})
            if isinstance(details, dict):
                details["interrupted_reason"] = (
                    "prompt file has no live worker event"
                    if prompt_only
                    else "worker started before the latest resume and never completed"
                )


def candidate_ids_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        candidate_id = value.get("id") or value.get("candidate")
        return [str(candidate_id)] if candidate_id else []
    if isinstance(value, (list, tuple)):
        candidate_ids: list[str] = []
        for item in value:
            candidate_ids.extend(candidate_ids_from_value(item))
        return candidate_ids
    return []


def has_renderable_incoming_edge(
    renderable_incoming_targets: set[str],
    target: str,
) -> bool:
    return target in renderable_incoming_targets


def add_prompt_only_worker_nodes(
    run_dir: Path,
    nodes: dict[str, dict[str, Any]],
    node_order: list[str],
    edges: list[dict[str, Any]],
    edge_keys: set[tuple[str, str]],
    edge_targets_by_source: dict[str, set[str]],
    renderable_incoming_targets: set[str],
    file_cache: dict[tuple[str, str | None], dict[str, Any] | None],
) -> None:
    worker_dir = run_dir / "workers"
    worker_dir_text = os.fspath(worker_dir)
    if not os.path.isdir(worker_dir_text):
        return

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edge_targets_by_source.setdefault(source, set()).add(target)
        if source in nodes:
            renderable_incoming_targets.add(target)
        edges.append({"from": source, "to": target})

    try:
        with os.scandir(worker_dir_text) as scanner:
            prompt_entries = sorted(
                (entry.name, entry.path)
                for entry in scanner
                if entry.name.endswith(".prompt.md") and entry.is_file(follow_symlinks=False)
            )
    except OSError:
        return
    for prompt_name, prompt_path_text in prompt_entries:
        candidate_id = prompt_name.removesuffix(".prompt.md")
        node_id = worker_node_id(candidate_id)
        if node_id in nodes:
            continue
        response_path_text = os.path.join(worker_dir_text, f"{candidate_id}.response.txt")
        status = "done" if os.path.isfile(response_path_text) else "running"
        file_items = existing_files(prompt_path_text, response_path_text, cache=file_cache)
        nodes[node_id] = {
            "id": node_id,
            "type": "worker",
            "title": worker_title(candidate_id, "worker"),
            "status": status,
            "body": read_text_preview(prompt_path_text),
            "meta": {"candidate": candidate_id, "source": "worker prompt file"},
            "details": {"prompt_path": prompt_path_text},
            "files": file_items,
        }
        node_order.append(node_id)
        renderable_incoming_targets.update(edge_targets_by_source.get(node_id, ()))
        if not has_renderable_incoming_edge(renderable_incoming_targets, node_id):
            add_edge("user", node_id)


def existing_files(
    *paths: str | os.PathLike[str],
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    return [item for path in paths if (item := file_item(path, cache=cache)) is not None]


def files_from_payload(
    payload: dict[str, Any],
    *,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    return files_from_refs(payload_file_refs(payload), cache=cache)


def payload_file_refs(payload: dict[str, Any], *, include_artifacts: bool = False) -> tuple[PayloadFileRef, ...]:
    refs: list[PayloadFileRef] = []
    for key, label in (
        ("prompt_path", "prompt"),
        ("response_path", "response"),
    ):
        path_text = payload.get(key)
        if isinstance(path_text, str) and path_text:
            refs.append(PayloadFileRef(path_text, label))
    if include_artifacts:
        for artifact in payload.get("artifacts", []) or []:
            if isinstance(artifact, str) and artifact:
                refs.append(PayloadFileRef(artifact, "artifact"))
    return tuple(refs)


def files_from_refs(
    refs: tuple[PayloadFileRef, ...],
    *,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for ref in refs:
        item = file_item(ref.path, label=ref.label, cache=cache)
        if item:
            files.append(item)
    return files


def files_from_candidate(
    candidate_dir: str | os.PathLike[str],
    candidate: dict[str, Any],
    *,
    candidate_file_previews: dict[str, str] | None = None,
    candidate_file_models: dict[str, CandidateFileModel] | None = None,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    candidate_id = str(candidate.get("id", ""))
    candidate_path_text = os.path.join(os.fspath(candidate_dir), f"{candidate_id}.json") if candidate_id else ""
    model = candidate_file_models.get(candidate_path_text) if candidate_file_models is not None and candidate_path_text else None
    files = files_from_refs(
        model.payload_refs if model is not None else payload_file_refs(candidate, include_artifacts=True),
        cache=cache,
    )
    if candidate_id:
        if model is not None:
            item = dict(model.candidate_item)
            if cache is not None:
                cache[(candidate_path_text, "candidate.json")] = item
        else:
            item = file_item(
                candidate_path_text,
                label="candidate.json",
                preview=(candidate_file_previews or {}).get(candidate_path_text),
                cache=cache,
            )
        if item:
            files.append(item)
    return files


def file_item(
    path: str | os.PathLike[str],
    *,
    label: str | None = None,
    preview: str | None = None,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    path_text = os.fspath(path)
    cache_key = (path_text, label)
    if cache is not None and cache_key in cache:
        item = cache[cache_key]
        return dict(item) if item is not None else None
    try:
        path_stat = os.stat(path_text)
    except OSError:
        if cache is not None:
            cache[cache_key] = None
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        if cache is not None:
            cache[cache_key] = None
        return None
    item = {
        "label": label or os.path.basename(path_text),
        "path": path_text,
        "size": path_stat.st_size,
        "preview": preview if preview is not None else read_text_preview(path_text, path_stat=path_stat),
    }
    if cache is not None:
        cache[cache_key] = item
    return dict(item)


def read_text_preview(path: str | os.PathLike[str], *, path_stat: os.stat_result | None = None) -> str:
    path_text = os.fspath(path)
    if path_stat is None:
        try:
            path_stat = os.stat(path_text)
        except OSError:
            return ""
        if not stat.S_ISREG(path_stat.st_mode):
            return ""

    cache_key = path_text
    cache_signature = (path_stat.st_mtime_ns, path_stat.st_ctime_ns, path_stat.st_size)
    with _TEXT_PREVIEW_CACHE_LOCK:
        cached = _TEXT_PREVIEW_CACHE.get(cache_key)
        if cached is not None and cached[:3] == cache_signature:
            return cached[3]

    try:
        with open(path_text, "rb") as handle:
            raw = handle.read(MAX_TEXT_PREVIEW_CHARS)
    except OSError:
        return ""
    preview = raw.decode("utf-8", errors="replace")
    with _TEXT_PREVIEW_CACHE_LOCK:
        if len(_TEXT_PREVIEW_CACHE) >= MAX_TEXT_PREVIEW_CACHE_ITEMS:
            _TEXT_PREVIEW_CACHE.clear()
        _TEXT_PREVIEW_CACHE[cache_key] = (*cache_signature, preview)
    return preview


def dedupe_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for file in files:
        path = str(file.get("path", ""))
        if not path or path in seen:
            continue
        seen.add(path)
        deduped.append(file)
    return deduped


def event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def manager_node_id(payload: dict[str, Any]) -> str:
    if payload.get("name"):
        return f"manager:{payload['name']}"
    return f"manager:manager_d{payload.get('depth', 0)}_{payload.get('parent') or 'root'}"


def diversity_node_id(payload: dict[str, Any]) -> str:
    if payload.get("name"):
        return f"diversity:{payload['name']}"
    return f"diversity:diversity_d{payload.get('depth', 0)}_{payload.get('parent') or 'root'}"


def worker_node_id(candidate_id: object) -> str:
    return f"worker:{candidate_id}"


def crow_node_id(payload: dict[str, Any]) -> str:
    if payload.get("name"):
        return f"crow:{payload['name']}"
    return f"crow:{payload.get('nudge_index', 0)}"


def resume_node_id(event_index: int) -> str:
    return f"resume:{event_index}"


def manager_title(payload: dict[str, Any]) -> str:
    return f"Manager d{payload.get('depth', 0)}"


def diversity_title(payload: dict[str, Any]) -> str:
    return f"Diversity d{payload.get('depth', 0)}"


def worker_title(candidate_id: str, label: str) -> str:
    return f"{label or 'worker'} ({candidate_id})"


def crow_title(payload: dict[str, Any]) -> str:
    return f"Crow {payload.get('nudge_index', 0)}"


def seed_label(seed: Any) -> str:
    if isinstance(seed, dict):
        return str(seed.get("label", "")).strip()
    return ""


def role_body(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = payload.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "\n\n".join(parts)


def short_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def node_depth(node: dict[str, Any]) -> int:
    meta = node.get("meta")
    if isinstance(meta, dict):
        try:
            return int(meta.get("depth"))
        except (TypeError, ValueError):
            pass
        candidate_id = str(meta.get("candidate", ""))
    else:
        candidate_id = ""
    if not candidate_id:
        node_id = str(node.get("id", ""))
        candidate_id = node_id.removeprefix("worker:")
    match = re.match(r"d(\d+)_w", candidate_id)
    return int(match.group(1)) if match else 0


def render_monitor_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pawahara Agent Monitor</title>
<style>
body { font-family: sans-serif; margin: 16px; }
button { margin: 2px 8px 2px 0; }
#summary { margin: 12px 0; }
#layout { display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 16px; align-items: start; }
.tree { border-left: 1px solid #aaa; margin-left: 4px; padding-left: 14px; }
.tree-children { border-left: 1px solid #ccc; margin: 8px 0 0 14px; padding-left: 14px; }
.tree-leaf { margin: 4px 0; }
.tree-key { color: #555; font-weight: bold; }
.tree-branch { margin: 6px 0; }
.tree-branch > summary { cursor: pointer; }
.tree-viewport { width: 100%; max-height: 72vh; overflow: auto; border: 1px solid #bbb; background: #fafafa; }
.tree-svg { display: block; max-width: none; background: transparent; }
.svg-edge { fill: none; stroke: #222; stroke-width: 2; }
.svg-edge.path-highlight { stroke: #e18a00; stroke-width: 5; }
.svg-node rect { stroke: #222; stroke-width: 2; rx: 8; }
.svg-node text { font-family: sans-serif; font-size: 12px; pointer-events: none; }
.svg-node.path-highlight rect { stroke: #e18a00; stroke-width: 4; }
.svg-node.selected rect { stroke-width: 4; }
.svg-title { font-weight: bold; font-size: 13px; }
.svg-meta { fill: #555; }
.svg-label { box-sizing: border-box; width: 100%; height: 100%; overflow: hidden; padding: 0 2px; font-family: sans-serif; color: #111; pointer-events: none; }
.svg-label-title { font-weight: bold; font-size: 13px; line-height: 16px; max-height: 32px; overflow: hidden; overflow-wrap: anywhere; word-break: break-all; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.svg-label-line { margin-top: 7px; color: #555; font-size: 12px; line-height: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.node { border: 1px solid #222; margin: 6px 0; padding: 10px; background: #fff; cursor: pointer; }
.node.selected { outline: 3px solid #444; }
.node.user { background: #fff8d8; }
.node.manager { background: #eef3ff; }
.node.diversity { background: #edf9ef; }
.node.worker { background: #f5f5f5; }
.node.crow { background: #fff0f0; }
.node.resume { background: #fff2b8; }
.node.stale { background: #eeeeee; color: #666; }
.node.running { outline: 3px solid #222; }
.node.path-highlight { border-color: #e18a00; box-shadow: inset 5px 0 #e18a00; }
.title { font-weight: bold; margin-bottom: 6px; }
.meta { color: #555; font-size: 12px; margin-bottom: 8px; }
.body { white-space: pre-wrap; max-height: 180px; overflow: auto; }
.file-list { margin-top: 8px; }
pre { white-space: pre-wrap; overflow: auto; border: 1px solid #ccc; padding: 8px; }
#eventTree { max-height: 420px; overflow: auto; }
#runState, #details, #fileContent, #allFiles { max-height: 360px; overflow: auto; }
</style>
</head>
<body>
<h1>Pawahara Agent Monitor</h1>
<div>
  <button onclick="refresh()">Refresh</button>
  <label><input id="auto" type="checkbox" checked> auto</label>
</div>
<div id="summary">loading</div>
<div id="layout">
  <main>
    <h2>Agents</h2>
    <div id="treeViewport" class="tree-viewport"><svg id="treeSvg" class="tree-svg"></svg></div>
    <div id="agentTree" class="tree"></div>
    <h2>Events</h2>
    <div id="eventTree" class="tree"></div>
  </main>
  <aside>
    <h2>Run State</h2>
    <div id="runState" class="tree"></div>
    <h2>Selected</h2>
    <div id="fileButtons"></div>
    <div id="details" class="tree">select a node</div>
    <h2>File</h2>
    <pre id="fileContent">select a file</pre>
    <h2>All Run Files</h2>
    <div id="allFileButtons"></div>
    <div id="allFiles" class="tree"></div>
  </aside>
</div>
<script>
let currentData = null;
let selectedNodeId = null;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function render(data) {
  currentData = data;
  const summary = document.getElementById('summary');
  const treeSvg = document.getElementById('treeSvg');
  const agentTree = document.getElementById('agentTree');
  const eventTree = document.getElementById('eventTree');
  const runState = document.getElementById('runState');
  agentTree.innerHTML = '';
  if (!data.run) {
    summary.textContent = 'No run yet.';
    eventTree.innerHTML = '';
    runState.innerHTML = '';
    return;
  }
  summary.textContent = `${data.run.run_id}  status=${data.run.status}  candidates=${data.counts.candidates}  running=${data.counts.running}`;
  runState.innerHTML = objectTree({
    run: data.run,
    counts: data.counts,
    role_states: data.role_states,
    result: data.result,
    edges: data.edges,
  }, 'run');
  renderSvgTree(treeSvg, data.nodes || [], data.edges || []);
  agentTree.innerHTML = renderAgentForest(data.nodes || [], data.edges || []);
  eventTree.innerHTML = renderEventTree(data.events || []);
  renderAllFiles(data.files || []);
  if (selectedNodeId) selectNode(selectedNodeId, false);
}

function fileButton(file) {
  return `<button onclick="event.stopPropagation(); loadFile('${encodeURIComponent(file.path)}')">${esc(file.label || file.relative_path || file.path)} ${esc(file.size ?? '')}</button>`;
}

function selectNode(nodeId, rerender = true) {
  try {
    nodeId = decodeURIComponent(nodeId);
  } catch (_error) {
  }
  selectedNodeId = nodeId;
  const node = (currentData?.nodes || []).find(item => item.id === nodeId);
  if (!node) return;
  document.getElementById('fileButtons').innerHTML = (node.files || []).map(fileButton).join('');
  document.getElementById('details').innerHTML = objectTree({
    id: node.id,
    type: node.type,
    status: node.status,
    score: node.score,
    path_highlight: node.path_highlight,
    path_reasons: node.path_reasons,
    body: node.body,
    meta: node.meta,
    details: node.details,
    files: node.files,
  }, 'selected');
  if (rerender) render(currentData);
}

function renderAllFiles(files) {
  document.getElementById('allFileButtons').innerHTML = files.map(fileButton).join('');
  document.getElementById('allFiles').innerHTML = renderFileTree(files);
}

function renderAgentForest(nodes, edges) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map();
  const incoming = new Set();
  for (const edge of edges) {
    if (!byId.has(edge.from) || !byId.has(edge.to)) continue;
    if (!children.has(edge.from)) children.set(edge.from, []);
    children.get(edge.from).push(edge.to);
    incoming.add(edge.to);
  }
  sortTreeChildren(children, byId);
  const roots = nodes.filter(node => !incoming.has(node.id));
  const orderedRoots = roots.length ? roots : nodes;
  const seen = new Set();
  return orderedRoots.map(node => renderAgentNode(node.id, byId, children, seen)).join('');
}

function buildTreeIndex(nodes, edges) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map();
  const incoming = new Set();
  for (const edge of edges) {
    if (!byId.has(edge.from) || !byId.has(edge.to)) continue;
    if (!children.has(edge.from)) children.set(edge.from, []);
    children.get(edge.from).push(edge.to);
    incoming.add(edge.to);
  }
  sortTreeChildren(children, byId);
  const roots = nodes.filter(node => !incoming.has(node.id));
  return { byId, children, roots: roots.length ? roots : nodes };
}

function sortTreeChildren(children, byId) {
  for (const childIds of children.values()) {
    childIds.sort((left, right) => nodePriority(byId.get(left)) - nodePriority(byId.get(right)));
  }
}

function nodePriority(node) {
  if (!node) return 99;
  if (node.status === 'running') return 0;
  if (node.type === 'resume') return 1;
  if (node.stale) return 8;
  if (node.status === 'blocked' || node.status === 'dead_end') return 9;
  return 4;
}

function renderSvgTree(svg, nodes, edges) {
  svg.innerHTML = '';
  const { byId, children, roots } = buildTreeIndex(nodes, edges);
  const positions = new Map();
  let row = 0;
  let maxDepth = 0;
  const seen = new Set();
  function visit(nodeId, depth) {
    if (seen.has(nodeId)) return;
    seen.add(nodeId);
    maxDepth = Math.max(maxDepth, depth);
    const childIds = children.get(nodeId) || [];
    if (!childIds.length) {
      positions.set(nodeId, { depth, row: row++ });
      return;
    }
    for (const childId of childIds) visit(childId, depth + 1);
    const childRows = childIds.map(id => positions.get(id)?.row).filter(value => value !== undefined);
    const average = childRows.length ? childRows.reduce((a, b) => a + b, 0) / childRows.length : row++;
    positions.set(nodeId, { depth, row: average });
  }
  for (const root of roots) visit(root.id, 0);
  for (const node of nodes) if (!positions.has(node.id)) visit(node.id, 0);

  const nodeWidth = 260;
  const nodeHeight = 118;
  const xGap = 360;
  const yGap = 180;
  const margin = 40;
  const width = Math.max(720, margin * 2 + maxDepth * xGap + nodeWidth);
  const height = Math.max(360, margin * 2 + Math.max(1, row) * yGap);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.style.width = `${width}px`;
  svg.style.height = `${height}px`;
  const point = id => {
    const pos = positions.get(id) || { depth: 0, row: 0 };
    return {
      x: margin + pos.depth * xGap,
      y: margin + pos.row * yGap,
    };
  };

  function appendEdge(edge) {
    if (!positions.has(edge.from) || !positions.has(edge.to)) return;
    const from = point(edge.from);
    const to = point(edge.to);
    const x1 = from.x + nodeWidth;
    const y1 = from.y + nodeHeight / 2;
    const x2 = to.x;
    const y2 = to.y + nodeHeight / 2;
    const mid = (x1 + x2) / 2;
    svg.appendChild(svgEl('path', {
      class: `svg-edge ${edge.path_highlight ? 'path-highlight' : ''}`,
      d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
    }));
  }
  for (const edge of edges.filter(edge => !edge.path_highlight)) appendEdge(edge);
  for (const edge of edges.filter(edge => edge.path_highlight)) appendEdge(edge);

  for (const node of nodes) {
    const pos = point(node.id);
    const group = svgEl('g', {
      class: `svg-node ${node.path_highlight ? 'path-highlight' : ''} ${node.id === selectedNodeId ? 'selected' : ''}`,
      tabindex: '0',
    });
    group.addEventListener('click', () => selectNode(node.id));
    const tooltip = svgEl('title', {});
    tooltip.textContent = `${node.title || node.id}\n${node.type || ''} ${node.status || ''}`;
    group.appendChild(tooltip);
    group.appendChild(svgEl('rect', {
      x: pos.x,
      y: pos.y,
      width: nodeWidth,
      height: nodeHeight,
      fill: svgNodeFill(node),
    }));
    group.appendChild(svgNodeLabel(node, pos.x + 10, pos.y + 8, nodeWidth - 20, nodeHeight - 14));
    svg.appendChild(group);
  }
}

function svgNodeFill(node) {
  if (node.stale) return '#eeeeee';
  if (node.status === 'solved') return '#c9f6cf';
  if (node.status === 'blocked' || node.status === 'dead_end') return '#eeeeee';
  if (node.type === 'user') return '#fff8d8';
  if (node.type === 'manager') return '#eef3ff';
  if (node.type === 'diversity') return '#edf9ef';
  if (node.type === 'crow') return '#fff0f0';
  if (node.type === 'resume') return '#fff2b8';
  if (node.status === 'running') return '#fff2b8';
  return '#f5f5f5';
}

function svgEl(name, attrs) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs || {})) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function svgNodeLabel(node, x, y, width, height) {
  const foreignObject = svgEl('foreignObject', { x, y, width, height });
  const container = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
  container.setAttribute('class', 'svg-label');

  const title = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
  title.setAttribute('class', 'svg-label-title');
  title.textContent = String(node.title || node.id || '');
  container.appendChild(title);

  const meta = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
  meta.setAttribute('class', 'svg-label-line');
  meta.textContent = `${node.type || ''}  ${node.status || ''}`;
  container.appendChild(meta);

  const body = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
  body.setAttribute('class', 'svg-label-line');
  body.textContent = node.score === undefined ? String(node.body || '') : `score ${node.score}`;
  container.appendChild(body);

  const id = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
  id.setAttribute('class', 'svg-label-line');
  id.textContent = String(node.id || '');
  container.appendChild(id);

  foreignObject.appendChild(container);
  return foreignObject;
}

function renderAgentNode(id, byId, children, seen) {
  const node = byId.get(id);
  if (!node || seen.has(id)) return '';
  seen.add(id);
  const childHtml = (children.get(id) || []).map(child => renderAgentNode(child, byId, children, seen)).join('');
  const score = node.score === undefined ? '' : ` score=${node.score}`;
  const encodedId = encodeURIComponent(node.id);
  return `
    <details class="tree-branch agent-branch" open>
      <summary>
        <span class="tree-key">${esc(node.title || node.id)}</span>
        <span>${esc(node.type)} status=${esc(node.status)}${esc(score)}</span>
      </summary>
      <div class="node ${esc(node.type || '')} ${node.status === 'running' ? 'running' : ''} ${node.path_highlight ? 'path-highlight' : ''} ${node.stale ? 'stale' : ''} ${node.id === selectedNodeId ? 'selected' : ''}" onclick="selectNode('${encodedId}')">
        <div class="title">${esc(node.title || node.id)}</div>
        <div class="meta">${esc(node.type)} status=${esc(node.status)}${esc(score)}</div>
        <div class="body">${esc(node.body || '')}</div>
        <div class="file-list">${(node.files || []).map(fileButton).join('')}</div>
      </div>
      <div class="tree-children">${childHtml}</div>
    </details>
  `;
}

function renderEventTree(events) {
  return events.map((event, index) => objectTree(event, `${index}: ${event.kind || 'event'}`)).join('');
}

function renderFileTree(files) {
  const root = {};
  for (const file of files) {
    const parts = String(file.relative_path || file.path || '').split('/').filter(Boolean);
    let cursor = root;
    for (const part of parts) {
      cursor.children ||= {};
      cursor.children[part] ||= {};
      cursor = cursor.children[part];
    }
    cursor.file = file;
  }
  return renderFileBranch('files', root);
}

function renderFileBranch(name, node) {
  const children = node.children || {};
  const childHtml = Object.keys(children).sort().map(key => renderFileBranch(key, children[key])).join('');
  if (node.file) {
    return `
      <details class="tree-branch">
        <summary>${esc(name)} ${esc(node.file.size)} bytes</summary>
        <div class="tree-children">
          ${fileButton(node.file)}
          ${objectTree(node.file, 'file')}
        </div>
      </details>
    `;
  }
  return `
    <details class="tree-branch" open>
      <summary>${esc(name)}</summary>
      <div class="tree-children">${childHtml}</div>
    </details>
  `;
}

function objectTree(value, label) {
  if (value === null || typeof value !== 'object') {
    return `<div class="tree-leaf"><span class="tree-key">${esc(label)}:</span> ${esc(value)}</div>`;
  }
  const isArray = Array.isArray(value);
  const entries = isArray ? value.map((item, index) => [String(index), item]) : Object.entries(value);
  const summary = `${label} ${isArray ? '[' + value.length + ']' : '{' + entries.length + '}'}`;
  return `
    <details class="tree-branch" open>
      <summary>${esc(summary)}</summary>
      <div class="tree-children">
        ${entries.map(([key, item]) => objectTree(item, key)).join('')}
      </div>
    </details>
  `;
}

async function loadFile(encodedPath) {
  const res = await fetch(`/api/file?path=${encodedPath}`, { cache: 'no-store' });
  const data = await res.json();
  if (!data.ok) {
    document.getElementById('fileContent').textContent = data.error || 'failed';
    return;
  }
  document.getElementById('fileContent').textContent = `${data.path}\\n${data.size} bytes\\n\\n${data.content}`;
}

async function refresh() {
  try {
    const res = await fetch('/api/latest', { cache: 'no-store' });
    render(await res.json());
  } catch (error) {
    document.getElementById('summary').textContent = String(error);
  }
}
setInterval(() => {
  if (document.getElementById('auto').checked) refresh();
}, 1000);
refresh();
</script>
</body>
</html>
"""
