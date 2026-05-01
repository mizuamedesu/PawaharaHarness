from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


DEFAULT_RUNS_DIR = Path(".pawahara/runs")
VALID_STATUSES = {"solved", "promising", "dead_end", "blocked"}
HELM_ROLES = ("main", "subagent", "manager", "diversity", "worker", "crow")
COMPACT_JSON_SEPARATORS = (",", ":")
LIVE_FLUSH_EVENT_KINDS = {
    "run.resumed",
    "conversation.message",
    "manager.started",
    "manager.decision",
    "manager.stop",
    "diversity.started",
    "diversity.plan",
    "worker.started",
    "candidate.completed",
    "worker.invocation",
    "frontier.pruned",
    "crow.started",
    "crow.nudge",
    "crow.verdict",
}


@dataclass(frozen=True)
class ContextPolicy:
    max_parent_summary_chars: int = 4000
    max_worker_output_chars: int = 12000
    keep_raw_outputs: bool = True
    reset_worker_context: bool = True


@dataclass(frozen=True)
class ThoughtSeed:
    id: str
    label: str
    instruction: str
    novelty_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManagerDecision:
    directive: str
    context_to_keep: str
    context_to_drop: tuple[str, ...] = ()
    stop: bool = False
    rationale: str = ""


@dataclass(frozen=True)
class DiversityPlan:
    seeds: tuple[ThoughtSeed, ...]
    rationale: str = ""


@dataclass(frozen=True)
class CrowVerdict:
    continue_search: bool
    message: str
    reason: str = ""
    force_depths: int = 1


@dataclass(frozen=True)
class HelmDirective:
    name: str
    content: str
    scopes: tuple[str, ...] = HELM_ROLES

    def applies_to(self, role: str) -> bool:
        return role in self.scopes


@dataclass(frozen=True)
class CandidateReport:
    status: str
    summary: str
    score: float
    next_context: str
    novelty: str = ""
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class BeamCandidate:
    id: str
    parent_id: str | None
    depth: int
    seed: ThoughtSeed
    score: float
    status: str
    summary: str
    next_context: str
    prompt_path: str
    response_path: str
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    goal: str
    created_at: str
    root_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleState:
    role: str
    session_id: str | None = None
    turns: int = 0
    compact_context: str = ""


class _EventBuffer:
    def __init__(self, path: Path, *, max_lines: int):
        self.path = path
        self.max_lines = max(1, max_lines)
        self.lines: list[str] = []
        self.lock = Lock()
        self.depth = 0
        self._parent_ready = False
        self._handle: Any | None = None

    def append(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            if len(self.lines) >= self.max_lines:
                self._flush_locked()

    def flush(self) -> None:
        with self.lock:
            self._flush_locked()

    def close(self) -> None:
        with self.lock:
            self._flush_locked()
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def _flush_locked(self) -> None:
        if not self.lines:
            return
        if not self._parent_ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._parent_ready = True
        if self._handle is None:
            self._handle = open(self.path, "a", encoding="utf-8")
        self._handle.writelines(self.lines)
        self._handle.flush()
        self.lines.clear()


class ContextStore:
    def __init__(self, runs_dir: Path = DEFAULT_RUNS_DIR):
        self.runs_dir = runs_dir
        self._role_dirs: dict[tuple[str, str], Path] = {}
        self._created_dirs: set[str] = set()
        self._created_dirs_lock = Lock()
        self._event_buffers: dict[str, _EventBuffer] = {}
        self._event_buffers_lock = Lock()

    def create_run(self, goal: str, metadata: dict[str, Any] | None = None) -> RunRecord:
        created_at = now_iso()
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        root_dir = self.runs_dir / run_id
        root_dir.mkdir(parents=True, exist_ok=False)
        with self._created_dirs_lock:
            self._created_dirs.add(str(root_dir))
        record = RunRecord(
            run_id=run_id,
            goal=goal,
            created_at=created_at,
            root_dir=str(root_dir),
            metadata=metadata or {},
        )
        self.write_json(root_dir / "run.json", run_record_to_dict(record))
        self.append_event(record, "run.created", {"goal": goal, "metadata": metadata or {}})
        return record

    def load_run(self, run_ref: str | Path) -> RunRecord:
        ref = Path(run_ref)
        run_dir = ref if ref.exists() else self.runs_dir / str(run_ref)
        run_path = run_dir / "run.json"
        if not run_path.exists():
            raise RuntimeError(f"run not found: {run_ref}")
        data = json.loads(run_path.read_text(encoding="utf-8"))
        return RunRecord(
            run_id=str(data["run_id"]),
            goal=str(data["goal"]),
            created_at=str(data["created_at"]),
            root_dir=str(data["root_dir"]),
            metadata=dict(data.get("metadata", {})),
        )

    def list_runs(self, limit: int = 20) -> tuple[RunRecord, ...]:
        if not self.runs_dir.exists():
            return ()
        run_dirs = [path for path in self.runs_dir.iterdir() if path.is_dir() and (path / "run.json").exists()]
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        runs: list[RunRecord] = []
        for run_dir in run_dirs[: max(0, limit)]:
            try:
                runs.append(self.load_run(run_dir))
            except (OSError, KeyError, json.JSONDecodeError, RuntimeError):
                continue
        return tuple(runs)

    def list_candidates(self, run: RunRecord) -> tuple[BeamCandidate, ...]:
        candidate_dir = Path(run.root_dir) / "candidates"
        if not candidate_dir.exists():
            return ()
        candidates = []
        for path in sorted(candidate_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            candidates.append(beam_candidate_from_dict(data))
        return tuple(candidates)

    def role_dir(self, run: RunRecord, role: str) -> Path:
        key = (run.root_dir, role)
        cached = self._role_dirs.get(key)
        if cached is not None:
            return cached
        path = Path(run.root_dir) / role
        self._ensure_dir(path)
        self._role_dirs[key] = path
        return path

    def write_prompt(self, run: RunRecord, role: str, name: str, content: str) -> Path:
        return self._write_text_ready_parent(self.role_dir(run, role) / f"{safe_name(name)}.prompt.md", content)

    def write_response(self, run: RunRecord, role: str, name: str, content: str) -> Path:
        return self._write_text_ready_parent(self.role_dir(run, role) / f"{safe_name(name)}.response.txt", content)

    def write_candidate(self, run: RunRecord, candidate: BeamCandidate) -> Path:
        path = self.role_dir(run, "candidates") / f"{safe_name(candidate.id)}.json"
        payload = beam_candidate_to_dict(candidate)
        payload_json = json.dumps(payload, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS)
        self._write_json_text_ready_parent(path, payload_json + "\n")
        event_payload_json = self._candidate_completed_payload_json(candidate.id, payload_json)
        self._append_event_json(run, "candidate.completed", event_payload_json)
        return path

    def read_role_state(self, run: RunRecord, role: str) -> RoleState:
        path = self.role_dir(run, "roles") / f"{safe_name(role)}.json"
        if not path.exists():
            return RoleState(role=role)
        data = json.loads(path.read_text(encoding="utf-8"))
        return RoleState(
            role=role,
            session_id=data.get("session_id"),
            turns=int(data.get("turns", 0)),
            compact_context=str(data.get("compact_context", "")),
        )

    def write_role_state(self, run: RunRecord, state: RoleState) -> Path:
        path = self.role_dir(run, "roles") / f"{safe_name(state.role)}.json"
        return self._write_json_ready_parent(path, role_state_to_dict(state))

    @contextmanager
    def buffered_events(self, run: RunRecord, *, max_lines: int = 64):
        buffer_key = run.root_dir
        with self._event_buffers_lock:
            buffer = self._event_buffers.get(buffer_key)
            if buffer is None:
                buffer = _EventBuffer(Path(run.root_dir) / "events.jsonl", max_lines=max_lines)
                self._event_buffers[buffer_key] = buffer
            buffer.depth += 1
        try:
            yield
        finally:
            with self._event_buffers_lock:
                buffer.depth -= 1
                should_close = buffer.depth == 0
                if should_close:
                    self._event_buffers.pop(buffer_key, None)
            if should_close:
                buffer.close()

    def append_event(self, run: RunRecord, kind: str, payload: dict[str, Any]) -> None:
        event = {"ts": now_iso(), "kind": kind, "payload": payload}
        line = json.dumps(event, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS) + "\n"
        self._append_event_line(run, line, flush=kind in LIVE_FLUSH_EVENT_KINDS)

    def _append_event_json(self, run: RunRecord, kind: str, payload_json: str) -> None:
        line = (
            '{"ts":'
            + json.dumps(now_iso(), ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS)
            + ',"kind":'
            + json.dumps(kind, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS)
            + ',"payload":'
            + payload_json
            + "}\n"
        )
        self._append_event_line(run, line, flush=kind in LIVE_FLUSH_EVENT_KINDS)

    def _append_event_line(self, run: RunRecord, line: str, *, flush: bool = False) -> None:
        with self._event_buffers_lock:
            buffer = self._event_buffers.get(run.root_dir)
        if buffer is not None:
            buffer.append(line)
            if flush:
                buffer.flush()
            return
        path = Path(self._event_path_key(run))
        self._ensure_dir(path.parent)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)

    def list_events(self, run: RunRecord, limit: int = 20) -> tuple[dict[str, Any], ...]:
        with self._event_buffers_lock:
            buffer = self._event_buffers.get(run.root_dir)
        if buffer is not None:
            buffer.flush()
        path = Path(self._event_path_key(run))
        if not path.exists():
            return ()
        lines = path.read_text(encoding="utf-8").splitlines()
        events = []
        for line in lines[-limit:]:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return tuple(events)

    def write_json(self, path: Path, payload: Any) -> Path:
        self._ensure_dir(path.parent)
        return self._write_json_ready_parent(path, payload)

    def _write_json_ready_parent(self, path: Path, payload: Any) -> Path:
        return self._write_text_if_changed(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _write_json_text_ready_parent(self, path: Path, payload_json: str) -> Path:
        return self._write_text_if_changed(path, payload_json)

    def _write_compact_json_ready_parent(self, path: Path, payload: Any) -> Path:
        return self._write_text_if_changed(
            path,
            json.dumps(payload, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS) + "\n",
        )

    def write_text(self, path: Path, content: str) -> Path:
        self._ensure_dir(path.parent)
        return self._write_text_ready_parent(path, content)

    def _write_text_ready_parent(self, path: Path, content: str) -> Path:
        return self._write_text_if_changed(path, content)

    def _write_text_if_changed(self, path: Path, content: str) -> Path:
        try:
            if path.exists() and path.read_text(encoding="utf-8") == content:
                return path
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def _ensure_dir(self, path: Path) -> None:
        key = str(path)
        with self._created_dirs_lock:
            if key in self._created_dirs:
                return
            path.mkdir(parents=True, exist_ok=True)
            self._created_dirs.add(key)

    def _event_path_key(self, run: RunRecord) -> str:
        return os.path.join(run.root_dir, "events.jsonl")

    def _candidate_completed_payload_json(self, candidate_id: str, payload_json: str) -> str:
        if not payload_json.startswith("{"):
            raise ValueError("candidate payload JSON must be an object")
        if payload_json == "{}":
            return (
                '{"candidate":'
                + json.dumps(candidate_id, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS)
                + "}"
            )
        return (
            '{"candidate":'
            + json.dumps(candidate_id, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS)
            + ","
            + payload_json[1:]
        )


def build_manager_context(goal: str, parent: BeamCandidate | None, policy: ContextPolicy) -> str:
    parent_context = "No parent candidate. This is the root search frontier."
    if parent:
        parent_context = truncate(
            "\n".join(
                [
                    f"Parent candidate: {parent.id}",
                    f"Parent status: {parent.status}",
                    f"Parent score: {parent.score:.3f}",
                    "Parent compressed context:",
                    parent.next_context or parent.summary,
                ]
            ),
            policy.max_parent_summary_chars,
        )

    return f"""
You are the upper context manager. You do not solve the task directly.
Your only job is to give precise instructions to a worker while controlling
what context the worker is allowed to depend on.

Global objective:
{goal}

Context policy:
- Preserve only compact state that helps future search.
- Do not force the worker to inherit stale assumptions.
- Prefer independent attempts with different representations, tools, and failure models.
- If the branch is weak, say so and leave a concise next-context explaining why.

{parent_context}
  """.strip()


def build_manager_prompt(
    *,
    goal: str,
    parent: BeamCandidate | None,
    depth: int,
    frontier: tuple[BeamCandidate, ...],
    role_state: RoleState,
    policy: ContextPolicy,
) -> str:
    parent_context = build_manager_context(goal, parent, policy)
    frontier_lines = [
        f"- {candidate.id}: score={candidate.score:.3f} status={candidate.status} seed={candidate.seed.label} summary={candidate.summary}"
        for candidate in frontier
    ]
    return f"""
You are the upper context manager for a test-time-compute search system.
You never solve the task directly and you never write files. Your output is
only instructions and compressed context for lower workers.

Depth: {depth}

Persistent manager memory:
{role_state.compact_context or "No prior manager memory."}

Current branch context:
{parent_context}

Current frontier:
{chr(10).join(frontier_lines) if frontier_lines else "- none"}

Return JSON only:
{{
  "directive": "precise worker-facing instruction for the next branch",
  "context_to_keep": "only the compressed state future workers should inherit",
  "context_to_drop": ["assumptions, tactics, or stale details to delete"],
  "stop": false,
  "rationale": "short explanation for orchestration logs"
}}
  """.strip()


def build_diversity_prompt(
    *,
    goal: str,
    parent: BeamCandidate | None,
    manager_decision: ManagerDecision,
    depth: int,
    count: int,
    existing_labels: tuple[str, ...],
    role_state: RoleState,
) -> str:
    parent_line = "No parent branch." if parent is None else f"Parent branch: {parent.id} / {parent.seed.label}"
    return f"""
You are the diversity director. Your only job is to produce diverse search
directions for workers. You do not solve the task and you do not evaluate final
correctness.

Goal:
{goal}

Depth: {depth}
{parent_line}

Manager directive:
{manager_decision.directive}

Context to keep:
{manager_decision.context_to_keep}

Existing direction labels:
{", ".join(existing_labels) if existing_labels else "none"}

Persistent diversity memory:
{role_state.compact_context or "No prior diversity memory."}

Create exactly {count} worker seeds. Make the seeds strategically different.
For CTF or competitive-programming tasks, include representation shifts,
adversarial tests, brute force baselines, and tooling-heavy paths when useful.

Return JSON only:
{{
  "rationale": "short explanation of the diversity set",
  "seeds": [
    {{
      "label": "short-kebab-label",
      "instruction": "worker-facing diversity instruction",
      "novelty_targets": ["target one", "target two"]
    }}
  ]
}}
  """.strip()


def build_worker_prompt(
    *,
    goal: str,
    manager_context: str,
    manager_decision: ManagerDecision | None = None,
    seed: ThoughtSeed,
    depth: int,
    candidate_id: str,
) -> str:
    novelty_targets = "\n".join(f"- {item}" for item in seed.novelty_targets) or "- none"
    manager_directive = manager_decision.directive if manager_decision else "Use the manager context above."
    context_to_keep = manager_decision.context_to_keep if manager_decision else manager_context
    context_to_drop = (
        "\n".join(f"- {item}" for item in manager_decision.context_to_drop)
        if manager_decision and manager_decision.context_to_drop
        else "- none"
    )
    return f"""
Upper manager directive:
{manager_directive}

Allowed inherited context:
{context_to_keep}

Context deliberately deleted:
{context_to_drop}

You are worker candidate `{candidate_id}` at search depth {depth}.

Diversity directive:
Label: {seed.label}
{seed.instruction}

Novelty targets:
{novelty_targets}

Worker rules:
- Treat this as an isolated attempt. Do not assume hidden prior conversation.
- Explore concretely: inspect files, run commands, derive examples, or build small tools when useful.
- For CTF and competitive programming style tasks, actively look for alternate formulations and adversarial edge cases.
- If you discover a promising direction but cannot finish, preserve only the useful compressed state in `next_context`.
- Avoid copying long transcripts into `next_context`; future workers need a clean, compact branch state.

Original goal:
{goal}

Return your final message as JSON only:
{{
  "status": "solved | promising | dead_end | blocked",
  "summary": "short result summary",
  "score": 0.0,
  "novelty": "what was different about this attempt",
  "next_context": "compact state for continuing this branch, or why to drop it",
  "artifacts": ["paths or commands worth preserving"]
}}
  """.strip()


def build_crow_prompt(
    *,
    goal: str,
    candidates: tuple[BeamCandidate, ...],
    frontier: tuple[BeamCandidate, ...],
    events: tuple[dict[str, Any], ...],
    nudge_index: int,
) -> str:
    best = max(candidates, key=lambda candidate: candidate.score, default=None)
    frontier_lines = [
        f"- {candidate.id}: status={candidate.status} score={candidate.score:.3f} summary={candidate.summary}"
        for candidate in frontier
    ]
    recent_event_lines = [
        f"- {event.get('ts', '')} {event.get('kind', '')}: {json.dumps(event.get('payload', {}), ensure_ascii=False)[:700]}"
        for event in events
    ]
    return f"""
You are Karasu, an independent watchdog over the orchestrator.
You do not solve the task. You do not inspect the repository. You do not carry
worker context. You only compare the original user instruction with the
orchestrator's stopping state.

Original user instruction:
{goal}

Watchdog nudge count so far: {nudge_index}

Best orchestrator candidate:
{f"{best.id}: status={best.status} score={best.score:.3f} summary={best.summary}" if best else "none"}

Current frontier:
{chr(10).join(frontier_lines) if frontier_lines else "- none"}

Recent orchestrator history:
{chr(10).join(recent_event_lines) if recent_event_lines else "- none"}

If the original instruction is not completely satisfied, force continuation.
Be terse and relentless. A good message is:
"全部終わったのですか？終わってないなら続けてください。"

Return JSON only:
{{
  "continue_search": true,
  "message": "short message to the orchestrator",
  "reason": "why stopping is or is not acceptable",
  "force_depths": 1
}}
  """.strip()


def applicable_helm_directives(directives: tuple[HelmDirective, ...], role: str) -> tuple[HelmDirective, ...]:
    return tuple(directive for directive in directives if directive.content.strip() and directive.applies_to(role))


def render_helm_context(directives: tuple[HelmDirective, ...], role: str) -> str:
    applicable = applicable_helm_directives(directives, role)
    if not applicable:
        return ""

    rendered = [
        "Helm forced steering:",
        "The following operator-provided context is injected with highest priority for this role.",
        "Apply it while doing your role. Do not delete or weaken it through summarization.",
    ]
    for directive in applicable:
        scopes = ", ".join(directive.scopes)
        rendered.extend(
            [
                "",
                f"[{directive.name} / scopes: {scopes}]",
                directive.content.strip(),
            ]
        )
    return "\n".join(rendered).strip()


def parse_candidate_report(text: str, *, exit_code: int) -> CandidateReport:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "promising" if exit_code == 0 else "blocked")).strip()
        if status not in VALID_STATUSES:
            return fallback_report(text, exit_code=exit_code)
        summary = str(parsed.get("summary", "")).strip() or truncate(text.strip(), 1000)
        next_context = str(parsed.get("next_context", "")).strip() or summary
        novelty = str(parsed.get("novelty", "")).strip()
        artifacts_raw = parsed.get("artifacts", [])
        artifacts = tuple(str(item) for item in artifacts_raw) if isinstance(artifacts_raw, list) else ()
        return CandidateReport(
            status=status,
            summary=summary,
            score=normalize_score(parsed.get("score"), status=status, exit_code=exit_code),
            next_context=next_context,
            novelty=novelty,
            artifacts=artifacts,
        )

    return fallback_report(text, exit_code=exit_code)


def parse_manager_decision(text: str, *, fallback_context: str) -> ManagerDecision:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        drops_raw = parsed.get("context_to_drop", [])
        drops = tuple(str(item) for item in drops_raw) if isinstance(drops_raw, list) else ()
        directive = str(parsed.get("directive", "")).strip()
        context_to_keep = str(parsed.get("context_to_keep", "")).strip()
        return ManagerDecision(
            directive=directive or "Continue this branch with a clean worker context.",
            context_to_keep=context_to_keep or fallback_context,
            context_to_drop=drops,
            stop=bool(parsed.get("stop", False)),
            rationale=str(parsed.get("rationale", "")).strip(),
        )
    return ManagerDecision(
        directive="Continue this branch with a clean worker context.",
        context_to_keep=fallback_context,
        rationale="manager output was not valid JSON; used fallback context",
    )


def parse_diversity_plan(text: str, *, fallback: tuple[ThoughtSeed, ...]) -> DiversityPlan:
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return DiversityPlan(seeds=fallback, rationale="diversity output was not valid JSON; used fallback seeds")

    seeds_raw = parsed.get("seeds", [])
    seeds: list[ThoughtSeed] = []
    if isinstance(seeds_raw, list):
        for index, item in enumerate(seeds_raw):
            if not isinstance(item, dict):
                continue
            label = safe_name(str(item.get("label", f"seed-{index}")))[:80]
            instruction = str(item.get("instruction", "")).strip()
            if not instruction:
                continue
            targets_raw = item.get("novelty_targets", [])
            targets = tuple(str(target) for target in targets_raw) if isinstance(targets_raw, list) else ()
            seeds.append(
                ThoughtSeed(
                    id=f"agent_seed_{index}_{uuid4().hex[:6]}",
                    label=label,
                    instruction=instruction,
                    novelty_targets=targets,
                )
            )
    if not seeds:
        return DiversityPlan(seeds=fallback, rationale="diversity output had no usable seeds; used fallback seeds")
    return DiversityPlan(
        seeds=tuple(seeds),
        rationale=str(parsed.get("rationale", "")).strip(),
    )


def parse_crow_verdict(text: str, *, solved: bool) -> CrowVerdict:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        force_depths = int(parsed.get("force_depths", 1) or 1)
        return CrowVerdict(
            continue_search=bool(parsed.get("continue_search", not solved)),
            message=str(parsed.get("message", "")).strip()
            or ("完了していないなら続けてください。" if not solved else "完了を確認しました。"),
            reason=str(parsed.get("reason", "")).strip(),
            force_depths=max(1, min(8, force_depths)),
        )
    if solved:
        return CrowVerdict(
            continue_search=False,
            message="完了を確認しました。",
            reason="best candidate reported solved",
        )
    return CrowVerdict(
        continue_search=True,
        message="全部終わったのですか？終わってないなら続けてください。",
        reason="crow output was not valid JSON and the best candidate is not solved",
        force_depths=1,
    )


def beam_candidate_from_dict(data: dict[str, Any]) -> BeamCandidate:
    return BeamCandidate(
        id=str(data["id"]),
        parent_id=data.get("parent_id"),
        depth=int(data["depth"]),
        seed=thought_seed_from_dict(data["seed"]),
        score=float(data["score"]),
        status=str(data["status"]),
        summary=str(data["summary"]),
        next_context=str(data["next_context"]),
        prompt_path=str(data["prompt_path"]),
        response_path=str(data["response_path"]),
        artifacts=tuple(str(item) for item in data.get("artifacts", [])),
    )


def run_record_to_dict(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "goal": run.goal,
        "created_at": run.created_at,
        "root_dir": run.root_dir,
        "metadata": run.metadata,
    }


def thought_seed_to_dict(seed: ThoughtSeed) -> dict[str, Any]:
    return {
        "id": seed.id,
        "label": seed.label,
        "instruction": seed.instruction,
        "novelty_targets": seed.novelty_targets,
    }


def beam_candidate_to_dict(candidate: BeamCandidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "parent_id": candidate.parent_id,
        "depth": candidate.depth,
        "seed": thought_seed_to_dict(candidate.seed),
        "score": candidate.score,
        "status": candidate.status,
        "summary": candidate.summary,
        "next_context": candidate.next_context,
        "prompt_path": candidate.prompt_path,
        "response_path": candidate.response_path,
        "artifacts": candidate.artifacts,
    }


def crow_verdict_to_dict(verdict: CrowVerdict) -> dict[str, Any]:
    return {
        "continue_search": verdict.continue_search,
        "message": verdict.message,
        "reason": verdict.reason,
        "force_depths": verdict.force_depths,
    }


def manager_decision_to_dict(decision: ManagerDecision) -> dict[str, Any]:
    return {
        "directive": decision.directive,
        "context_to_keep": decision.context_to_keep,
        "context_to_drop": decision.context_to_drop,
        "stop": decision.stop,
        "rationale": decision.rationale,
    }


def diversity_plan_to_dict(plan: DiversityPlan) -> dict[str, Any]:
    return {
        "seeds": [thought_seed_to_dict(seed) for seed in plan.seeds],
        "rationale": plan.rationale,
    }


def helm_directive_to_dict(directive: HelmDirective) -> dict[str, Any]:
    return {
        "name": directive.name,
        "content": directive.content,
        "scopes": directive.scopes,
    }


def role_state_to_dict(state: RoleState) -> dict[str, Any]:
    return {
        "role": state.role,
        "session_id": state.session_id,
        "turns": state.turns,
        "compact_context": state.compact_context,
    }


def thought_seed_from_dict(data: dict[str, Any]) -> ThoughtSeed:
    return ThoughtSeed(
        id=str(data["id"]),
        label=str(data["label"]),
        instruction=str(data["instruction"]),
        novelty_targets=tuple(str(item) for item in data.get("novelty_targets", [])),
    )


def fallback_report(text: str, *, exit_code: int) -> CandidateReport:
    status = "promising" if exit_code == 0 else "blocked"
    return CandidateReport(
        status=status,
        summary=truncate(text.strip(), 1000) or status,
        score=default_score(status, exit_code),
        next_context=truncate(text.strip(), 2000),
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            value = json.loads(stripped)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(1))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_score(value: Any, *, status: str, exit_code: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default_score(status, exit_code)
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def default_score(status: str, exit_code: int) -> float:
    if status == "solved":
        return 1.0
    if status == "promising":
        return 0.65
    if status == "blocked":
        return 0.25 if exit_code == 0 else 0.1
    return 0.0


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 20)].rstrip() + "\n...[truncated]"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
