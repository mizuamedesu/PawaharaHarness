from __future__ import annotations

import json
from pathlib import Path

from pawahara_harness.agents import AgentLaunchSpec, AgentResult
from pawahara_harness.context import parse_candidate_report, parse_diversity_plan, parse_manager_decision
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
        ),
    )
    loaded = first.store.load_run(first_result.run.run_id)
    resumed = second.run(goal=loaded.goal, command="agent", cwd=str(tmp_path), resume_run=loaded)

    assert len(first_result.candidates) == 2
    assert len(second_runtime.calls) == 2
    assert len(resumed.candidates) == 4
    assert max(candidate.depth for candidate in resumed.candidates) == 1


def test_agentic_roles_drive_manager_and_diversity(tmp_path: Path) -> None:
    runtime = RoleAwareRuntime()
    orchestrator = BeamSearchOrchestrator(
        runtime=runtime,
        config=SearchConfig(beam_width=1, branch_factor=2, max_depth=1, max_workers=1),
    )
    orchestrator.store.runs_dir = tmp_path / "runs"

    result = orchestrator.run(goal="solve puzzle", command="agent", cwd=str(tmp_path))

    assert [call.role for call in runtime.calls] == ["manager", "diversity", "worker"]
    assert result.best_candidate is not None
    assert result.best_candidate.seed.label == "graph-model"
    assert result.best_candidate.next_context == "graph state"
    assert Path(result.run.root_dir, "roles", "manager.json").exists()
    assert Path(result.run.root_dir, "roles", "diversity.json").exists()
