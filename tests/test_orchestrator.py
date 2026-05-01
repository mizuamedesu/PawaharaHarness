from __future__ import annotations

import json
from pathlib import Path

from pawahara_harness.agents import AgentLaunchSpec, AgentResult
from pawahara_harness.context import (
    ContextStore,
    HelmDirective,
    parse_candidate_report,
    parse_crow_verdict,
    parse_diversity_plan,
    parse_manager_decision,
)
from pawahara_harness.orchestrator import BeamSearchOrchestrator, DiversityDirector, SearchConfig


class ScoringRuntime:
    def __init__(self) -> None:
        self.calls: list[AgentLaunchSpec] = []

    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        self.calls.append(spec)
        score = 0.2 + (0.1 * len(self.calls))
        payload = {
            "status": "promising",
            "summary": f"candidate {len(self.calls)}",
            "score": score,
            "novelty": "different branch",
            "next_context": f"keep branch {len(self.calls)}",
            "artifacts": [],
        }
        return AgentResult(
            name=spec.name,
            role=spec.role,
            sandbox_id="fake",
            command=spec.command,
            stdout=json.dumps(payload),
            stderr="",
            exit_code=0,
        )


class RoleAwareRuntime:
    def __init__(self) -> None:
        self.calls: list[AgentLaunchSpec] = []

    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        self.calls.append(spec)
        if spec.role == "manager":
            stdout = json.dumps(
                {
                    "directive": "use a clean invariant branch",
                    "context_to_keep": "only invariant facts",
                    "context_to_drop": ["previous brute force guess"],
                    "stop": False,
                }
            )
        elif spec.role == "diversity":
            stdout = json.dumps(
                {
                    "rationale": "split representation",
                    "seeds": [
                        {
                            "label": "graph-model",
                            "instruction": "model the state as a graph",
                            "novelty_targets": ["graph"],
                        }
                    ],
                }
            )
        else:
            stdout = json.dumps(
                {
                    "status": "promising",
                    "summary": "worker used graph model",
                    "score": 0.8,
                    "next_context": "graph state",
                    "artifacts": [],
                }
            )
        return AgentResult(
            name=spec.name,
            role=spec.role,
            sandbox_id="fake",
            command=spec.command,
            stdout=stdout,
            stderr="",
            exit_code=0,
            session_id=f"{spec.role}-session",
        )


class CrowRuntime:
    def __init__(self) -> None:
        self.calls: list[AgentLaunchSpec] = []

    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        self.calls.append(spec)
        if spec.role == "crow":
            stdout = json.dumps(
                {
                    "continue_search": True,
                    "message": "全部終わったのですか？終わってないなら続けてください。",
                    "reason": "best candidate is not solved",
                    "force_depths": 1,
                }
            )
        else:
            stdout = json.dumps(
                {
                    "status": "promising",
                    "summary": f"worker {len([call for call in self.calls if call.role == 'worker'])}",
                    "score": 0.5,
                    "next_context": "continue",
                    "artifacts": [],
                }
            )
        return AgentResult(
            name=spec.name,
            role=spec.role,
            sandbox_id="fake",
            command=spec.command,
            stdout=stdout,
            stderr="",
            exit_code=0,
        )


class RaisingRuntime:
    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        raise RuntimeError(f"boom from {spec.name}")


def test_parse_candidate_report_reads_fenced_json() -> None:
    report = parse_candidate_report(
        '```json\n{"status":"solved","summary":"done","score":0.95,"next_context":"answer"}\n```',
        exit_code=0,
    )
    assert report.status == "solved"
    assert report.score == 0.95
    assert report.next_context == "answer"


def test_parse_manager_decision_reads_context_controls() -> None:
    decision = parse_manager_decision(
        '{"directive":"try invariant path","context_to_keep":"state A","context_to_drop":["old guess"],"stop":false}',
        fallback_context="fallback",
    )
    assert decision.directive == "try invariant path"
    assert decision.context_to_keep == "state A"
    assert decision.context_to_drop == ("old guess",)


def test_parse_diversity_plan_reads_agent_seeds() -> None:
    plan = parse_diversity_plan(
        '{"seeds":[{"label":"bitset","instruction":"model states as bitsets","novelty_targets":["encoding"]}]}',
        fallback=(),
    )
    assert plan.seeds[0].label == "bitset"
    assert plan.seeds[0].novelty_targets == ("encoding",)


def test_parse_crow_verdict_forces_continuation() -> None:
    verdict = parse_crow_verdict(
        '{"continue_search":true,"message":"続けてください","reason":"not done","force_depths":2}',
        solved=False,
    )
    assert verdict.continue_search
    assert verdict.force_depths == 2


def test_diversity_director_generates_distinct_seeds() -> None:
    seeds = DiversityDirector().seeds(depth=0, parent=None, count=4)
    assert len(seeds) == 4
    assert len({seed.label for seed in seeds}) == 4


def test_beam_search_orchestrator_prunes_and_persists(tmp_path: Path) -> None:
    runtime = ScoringRuntime()
    orchestrator = BeamSearchOrchestrator(
        runtime=runtime,
        config=SearchConfig(
            beam_width=1,
            branch_factor=2,
            max_depth=2,
            max_workers=1,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    result = orchestrator.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))

    assert len(runtime.calls) == 4
    assert len(result.candidates) == 4
    assert len(result.frontier) == 1
    assert result.best_candidate is not None
    assert result.best_candidate.score > 0.0
    assert Path(result.best_candidate.prompt_path).exists()
    assert Path(result.run.root_dir, "result.json").exists()


def test_search_persists_artifacts_and_per_candidate_event_order(tmp_path: Path) -> None:
    runtime = ScoringRuntime()
    orchestrator = BeamSearchOrchestrator(
        runtime=runtime,
        config=SearchConfig(
            beam_width=4,
            branch_factor=8,
            max_depth=1,
            max_workers=4,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    result = orchestrator.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))
    run_root = Path(result.run.root_dir)

    assert (run_root / "run.json").exists()
    assert (run_root / "events.jsonl").exists()
    assert (run_root / "result.json").exists()
    assert len(result.candidates) == 8
    for candidate in result.candidates:
        assert Path(candidate.prompt_path).exists()
        assert Path(candidate.response_path).exists()
        candidate_json = run_root / "candidates" / f"{candidate.id}.json"
        assert candidate_json.exists()
        assert json.loads(candidate_json.read_text(encoding="utf-8"))["id"] == candidate.id

    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "run.created"
    assert kinds[-1] == "frontier.pruned"

    positions: dict[tuple[str, str], int] = {}
    for index, event in enumerate(events):
        candidate_id = event.get("payload", {}).get("candidate")
        if candidate_id:
            positions[(candidate_id, event["kind"])] = index

    for candidate in result.candidates:
        assert positions[(candidate.id, "worker.started")] < positions[(candidate.id, "candidate.completed")]
        assert positions[(candidate.id, "candidate.completed")] < positions[(candidate.id, "worker.invocation")]


def test_buffered_events_flush_when_worker_runtime_raises(tmp_path: Path) -> None:
    orchestrator = BeamSearchOrchestrator(
        runtime=RaisingRuntime(),
        config=SearchConfig(
            beam_width=1,
            branch_factor=1,
            max_depth=1,
            max_workers=1,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    try:
        orchestrator.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))
    except RuntimeError as exc:
        assert "boom from" in str(exc)
    else:
        raise AssertionError("runtime failure should propagate")

    run_roots = list((tmp_path / "runs").iterdir())
    assert len(run_roots) == 1
    events = [
        json.loads(line)
        for line in (run_roots[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == ["run.created", "worker.started"]


def test_empty_response_artifact_preserves_path(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("empty responses")

    response = store.write_response(run, "workers", "a", "")

    assert response.exists()
    assert response.read_text(encoding="utf-8") == ""


def test_beam_search_orchestrator_resumes_from_saved_frontier(tmp_path: Path) -> None:
    first_runtime = ScoringRuntime()
    first = BeamSearchOrchestrator(
        runtime=first_runtime,
        config=SearchConfig(
            beam_width=1,
            branch_factor=2,
            max_depth=1,
            max_workers=1,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    first.store.runs_dir = tmp_path / "runs"
    first_result = first.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))

    second_runtime = ScoringRuntime()
    second = BeamSearchOrchestrator(
        runtime=second_runtime,
        store=first.store,
        config=SearchConfig(
            beam_width=1,
            branch_factor=2,
            max_depth=1,
            max_workers=1,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    loaded = first.store.load_run(first_result.run.run_id)
    resumed = second.run(goal=loaded.goal, command="agent", cwd=str(tmp_path), resume_run=loaded)

    assert len(first_result.candidates) == 2
    assert len(second_runtime.calls) == 2
    assert len(resumed.candidates) == 4
    assert max(candidate.depth for candidate in resumed.candidates) == 1


def test_resume_message_is_recorded_and_passed_to_workers(tmp_path: Path) -> None:
    first_runtime = ScoringRuntime()
    first = BeamSearchOrchestrator(
        runtime=first_runtime,
        config=SearchConfig(
            beam_width=1,
            branch_factor=1,
            max_depth=1,
            max_workers=1,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    first.store.runs_dir = tmp_path / "runs"
    first_result = first.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))

    second_runtime = ScoringRuntime()
    second = BeamSearchOrchestrator(
        runtime=second_runtime,
        store=first.store,
        config=SearchConfig(
            beam_width=1,
            branch_factor=1,
            max_depth=1,
            max_workers=1,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=False,
        ),
    )
    loaded = first.store.load_run(first_result.run.run_id)
    second.run(
        goal=loaded.goal,
        command="agent",
        cwd=str(tmp_path),
        resume_run=loaded,
        resume_message="try the parser branch now",
    )

    assert second_runtime.calls
    assert "Latest resume message from the user" in second_runtime.calls[0].prompt
    assert "try the parser branch now" in second_runtime.calls[0].prompt
    events = first.store.list_events(loaded, limit=50)
    assert any(event["kind"] == "conversation.message" for event in events)


def test_agentic_roles_drive_manager_and_diversity(tmp_path: Path) -> None:
    runtime = RoleAwareRuntime()
    orchestrator = BeamSearchOrchestrator(
        runtime=runtime,
        config=SearchConfig(beam_width=1, branch_factor=2, max_depth=1, max_workers=1, crow_enabled=False),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    result = orchestrator.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))

    assert [call.role for call in runtime.calls] == ["manager", "diversity", "worker"]
    assert result.best_candidate is not None
    assert result.best_candidate.seed.label == "graph-model"
    assert result.best_candidate.next_context == "graph state"
    assert Path(result.run.root_dir, "roles", "manager.json").exists()
    assert Path(result.run.root_dir, "roles", "diversity.json").exists()


def test_helm_steering_is_injected_into_scoped_worker_prompt(tmp_path: Path) -> None:
    runtime = ScoringRuntime()
    orchestrator = BeamSearchOrchestrator(
        runtime=runtime,
        config=SearchConfig(
            beam_width=1,
            branch_factor=1,
            max_depth=1,
            max_workers=1,
            agentic_roles=False,
            crow_enabled=False,
            helm_directives=(
                HelmDirective(
                    name="ctf-steer",
                    content="Always build a quick falsification harness before accepting a branch.",
                    scopes=("worker",),
                ),
            ),
        ),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    result = orchestrator.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))

    assert runtime.calls
    assert runtime.calls[0].role == "worker"
    assert "Helm forced steering" in runtime.calls[0].prompt
    assert "quick falsification harness" in runtime.calls[0].prompt
    assert result.best_candidate is not None
    prompt_text = Path(result.best_candidate.prompt_path).read_text(encoding="utf-8")
    assert "quick falsification harness" in prompt_text
    events = orchestrator.store.list_events(result.run, limit=20)
    assert any(event["kind"] == "helm.injected" for event in events)


def test_crow_watchdog_forces_additional_depth(tmp_path: Path) -> None:
    runtime = CrowRuntime()
    orchestrator = BeamSearchOrchestrator(
        runtime=runtime,
        config=SearchConfig(
            beam_width=1,
            branch_factor=1,
            max_depth=1,
            max_workers=1,
            stop_on_solved=False,
            agentic_roles=False,
            crow_enabled=True,
            crow_max_nudges=1,
        ),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    result = orchestrator.run(goal="solve all tasks", command="agent", cwd=str(tmp_path))

    assert [call.role for call in runtime.calls] == ["worker", "crow", "worker"]
    assert len(result.candidates) == 2
    assert result.crow_nudges[0].continue_search
    assert Path(result.run.root_dir, "crow", "crow_0.prompt.md").exists()
