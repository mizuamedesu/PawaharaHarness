from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_RUNS_DIR = Path(".pawahara/runs")
VALID_STATUSES = {"solved", "promising", "dead_end", "blocked"}


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


class ContextStore:
    def __init__(self, runs_dir: Path = DEFAULT_RUNS_DIR):
        self.runs_dir = runs_dir

    def create_run(self, goal: str, metadata: dict[str, Any] | None = None) -> RunRecord:
        created_at = now_iso()
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        root_dir = self.runs_dir / run_id
        root_dir.mkdir(parents=True, exist_ok=False)
        record = RunRecord(
            run_id=run_id,
            goal=goal,
            created_at=created_at,
            root_dir=str(root_dir),
            metadata=metadata or {},
        )
        self.write_json(root_dir / "run.json", asdict(record))
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
        path = Path(run.root_dir) / role
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_prompt(self, run: RunRecord, role: str, name: str, content: str) -> Path:
        return self.write_text(self.role_dir(run, role) / f"{safe_name(name)}.prompt.md", content)

    def write_response(self, run: RunRecord, role: str, name: str, content: str) -> Path:
        return self.write_text(self.role_dir(run, role) / f"{safe_name(name)}.response.txt", content)

    def write_candidate(self, run: RunRecord, candidate: BeamCandidate) -> Path:
        path = self.role_dir(run, "candidates") / f"{safe_name(candidate.id)}.json"
        self.write_json(path, asdict(candidate))
        self.append_event(run, "candidate.completed", asdict(candidate))
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
        return self.write_json(path, asdict(state))

    def append_event(self, run: RunRecord, kind: str, payload: dict[str, Any]) -> None:
        event = {"ts": now_iso(), "kind": kind, "payload": payload}
        path = Path(run.root_dir) / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def write_text(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


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
