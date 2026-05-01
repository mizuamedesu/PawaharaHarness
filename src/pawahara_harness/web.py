from __future__ import annotations

import json
import os
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
    candidates = read_candidate_files(run_dir / "candidates")
    result_data = read_json(run_dir / "result.json")
    role_states = read_role_states(run_dir / "roles")
    result_exists = (run_dir / "result.json").exists()
    nodes, edges = build_nodes_and_edges(run_dir, run_data, events, candidates, result_exists=result_exists)
    running = sum(1 for node in nodes if node.get("status") == "running")
    completed = sum(1 for node in nodes if node.get("type") == "worker" and node.get("status") != "running")
    run_status = "running" if running else "finished" if result_exists else "running"
    for node in nodes:
        if node.get("id") == "user":
            node["status"] = run_status
            break
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
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


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
    if not candidate_dir.exists():
        return []
    candidates: list[dict[str, Any]] = []
    for path in sorted(candidate_dir.glob("*.json")):
        data = read_json(path)
        if data:
            candidates.append(data)
    return candidates


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
    for dirpath, _dirnames, filenames in os.walk(run_dir_text):
        for filename in filenames:
            path_text = os.path.join(dirpath, filename)
            try:
                path_stat = os.stat(path_text)
            except OSError:
                continue
            files.append(
                {
                    "path": path_text,
                    "relative_path": os.path.relpath(path_text, run_dir_text),
                    "size": path_stat.st_size,
                    "mtime": path_stat.st_mtime,
                }
            )
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
    result_exists: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    edges: list[dict[str, str]] = []
    edge_keys: set[tuple[str, str]] = set()
    file_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}

    def upsert(node_id: str, **fields: Any) -> dict[str, Any]:
        node = nodes.setdefault(node_id, {"id": node_id})
        if node_id not in node_order:
            node_order.append(node_id)
        for key, value in fields.items():
            if value is not None:
                node[key] = value
        return node

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
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

    for event in events:
        kind = str(event.get("kind", ""))
        payload = event_payload(event)
        if kind == "manager.started":
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
        file_items.extend(files_from_candidate(run_dir, candidate, cache=file_cache))
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
        if has_renderable_incoming_edge(edges, nodes, key):
            continue
        if parent:
            add_edge(worker_node_id(parent), key)
        else:
            diversity_id = diversity_node_id(candidate)
            add_edge(diversity_id if diversity_id in nodes else "user", key)

    add_prompt_only_worker_nodes(run_dir, nodes, node_order, edges, edge_keys, file_cache)

    return [nodes[node_id] for node_id in node_order], edges


def has_renderable_incoming_edge(
    edges: list[dict[str, str]],
    nodes: dict[str, dict[str, Any]],
    target: str,
) -> bool:
    return any(edge.get("to") == target and edge.get("from") in nodes for edge in edges)


def add_prompt_only_worker_nodes(
    run_dir: Path,
    nodes: dict[str, dict[str, Any]],
    node_order: list[str],
    edges: list[dict[str, str]],
    edge_keys: set[tuple[str, str]],
    file_cache: dict[tuple[str, str | None], dict[str, Any] | None],
) -> None:
    worker_dir = run_dir / "workers"
    if not worker_dir.exists():
        return

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"from": source, "to": target})

    for prompt_path in sorted(worker_dir.glob("*.prompt.md")):
        candidate_id = prompt_path.name.removesuffix(".prompt.md")
        node_id = worker_node_id(candidate_id)
        if node_id in nodes:
            continue
        response_path = prompt_path.with_name(f"{candidate_id}.response.txt")
        status = "done" if response_path.exists() else "running"
        file_items = existing_files(prompt_path, response_path, cache=file_cache)
        nodes[node_id] = {
            "id": node_id,
            "type": "worker",
            "title": worker_title(candidate_id, "worker"),
            "status": status,
            "body": read_text_preview(prompt_path),
            "meta": {"candidate": candidate_id, "source": "worker prompt file"},
            "details": {"prompt_path": str(prompt_path)},
            "files": file_items,
        }
        node_order.append(node_id)
        if not has_renderable_incoming_edge(edges, nodes, node_id):
            add_edge("user", node_id)


def existing_files(
    *paths: Path,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    return [item for path in paths if (item := file_item(path, cache=cache)) is not None]


def files_from_payload(
    payload: dict[str, Any],
    *,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for key, label in (
        ("prompt_path", "prompt"),
        ("response_path", "response"),
    ):
        path_text = payload.get(key)
        if isinstance(path_text, str) and path_text:
            item = file_item(Path(path_text), label=label, cache=cache)
            if item:
                files.append(item)
    return files


def files_from_candidate(
    run_dir: Path,
    candidate: dict[str, Any],
    *,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    files = files_from_payload(candidate, cache=cache)
    candidate_id = str(candidate.get("id", ""))
    if candidate_id:
        item = file_item(run_dir / "candidates" / f"{candidate_id}.json", label="candidate.json", cache=cache)
        if item:
            files.append(item)
    for artifact in candidate.get("artifacts", []) or []:
        if isinstance(artifact, str) and artifact:
            item = file_item(Path(artifact), label="artifact", cache=cache)
            if item:
                files.append(item)
    return dedupe_files(files)


def file_item(
    path: Path,
    *,
    label: str | None = None,
    cache: dict[tuple[str, str | None], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    cache_key = (str(path), label)
    if cache is not None and cache_key in cache:
        item = cache[cache_key]
        return dict(item) if item is not None else None
    try:
        path_stat = path.stat()
    except OSError:
        if cache is not None:
            cache[cache_key] = None
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        if cache is not None:
            cache[cache_key] = None
        return None
    item = {
        "label": label or path.name,
        "path": str(path),
        "size": path_stat.st_size,
        "preview": read_text_preview(path),
    }
    if cache is not None:
        cache[cache_key] = item
    return dict(item)


def read_text_preview(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_TEXT_PREVIEW_CHARS)
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


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
.svg-node rect { stroke: #222; stroke-width: 2; rx: 8; }
.svg-node text { font-family: sans-serif; font-size: 12px; pointer-events: none; }
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
.node.running { outline: 3px solid #222; }
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
  const roots = nodes.filter(node => !incoming.has(node.id));
  return { byId, children, roots: roots.length ? roots : nodes };
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

  for (const edge of edges) {
    if (!positions.has(edge.from) || !positions.has(edge.to)) continue;
    const from = point(edge.from);
    const to = point(edge.to);
    const x1 = from.x + nodeWidth;
    const y1 = from.y + nodeHeight / 2;
    const x2 = to.x;
    const y2 = to.y + nodeHeight / 2;
    const mid = (x1 + x2) / 2;
    svg.appendChild(svgEl('path', {
      class: 'svg-edge',
      d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
    }));
  }

  for (const node of nodes) {
    const pos = point(node.id);
    const group = svgEl('g', {
      class: `svg-node ${node.id === selectedNodeId ? 'selected' : ''}`,
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
  if (node.status === 'solved') return '#c9f6cf';
  if (node.status === 'blocked' || node.status === 'dead_end') return '#f8caca';
  if (node.type === 'user') return '#fff8d8';
  if (node.type === 'manager') return '#eef3ff';
  if (node.type === 'diversity') return '#edf9ef';
  if (node.type === 'crow') return '#fff0f0';
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
      <div class="node ${esc(node.type || '')} ${node.status === 'running' ? 'running' : ''} ${node.id === selectedNodeId ? 'selected' : ''}" onclick="selectNode('${encodedId}')">
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
