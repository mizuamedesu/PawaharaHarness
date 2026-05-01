from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agents import AgentLaunchSpec, AgentResult, AgentRuntime
from .context import (
    BeamCandidate,
    ContextPolicy,
    ContextStore,
    CrowVerdict,
    DiversityPlan,
    HelmDirective,
    ManagerDecision,
    RoleState,
    RunRecord,
    ThoughtSeed,
    applicable_helm_directives,
    build_crow_prompt,
    build_manager_context,
    build_diversity_prompt,
    build_manager_prompt,
    build_worker_prompt,
    parse_candidate_report,
    parse_crow_verdict,
    parse_diversity_plan,
    parse_manager_decision,
    render_helm_context,
    truncate,
)


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 4
    branch_factor: int = 4
    max_depth: int = 2
    max_workers: int = 4
    stop_on_solved: bool = True
    agentic_roles: bool = True
    reuse_role_sessions: bool = True
    model: str | None = None
    effort: str | None = None
    crow_enabled: bool = True
    crow_max_nudges: int = 3
    crow_event_limit: int = 20
    helm_directives: tuple[HelmDirective, ...] = ()
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)


@dataclass(frozen=True)
class SearchResult:
    run: RunRecord
    best_candidate: BeamCandidate | None
    frontier: tuple[BeamCandidate, ...]
    candidates: tuple[BeamCandidate, ...]
    crow_nudges: tuple[CrowVerdict, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": asdict(self.run),
            "best_candidate": asdict(self.best_candidate) if self.best_candidate else None,
            "frontier": [asdict(candidate) for candidate in self.frontier],
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "crow_nudges": [asdict(nudge) for nudge in self.crow_nudges],
        }


class DiversityDirector:
    """Owns only diversity pressure, not task state or tool execution."""

    _templates: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        (
            "direct-construction",
            "Try the most direct constructive path. Build or modify the smallest concrete thing that could satisfy the objective.",
            ("minimal working path", "fast feedback", "baseline solution"),
        ),
        (
            "adversarial-edge-cases",
            "Attack assumptions. Search for counterexamples, boundary cases, hidden constraints, and places where a plausible solution fails.",
            ("negative tests", "edge cases", "failure modes"),
        ),
        (
            "formal-invariants",
            "Model the problem in terms of invariants, proofs, reductions, or necessary conditions before acting.",
            ("invariants", "proof obligations", "state representation"),
        ),
        (
            "instrumentation",
            "Prefer measurement. Add temporary probes, small scripts, traces, fuzzers, or differential checks to reveal structure.",
            ("observability", "experiments", "empirical ranking"),
        ),
        (
            "alternate-representation",
            "Re-encode the problem using a different representation: graph, algebra, automaton, bitset, SAT/SMT, dynamic programming, or protocol trace.",
            ("representation shift", "compression", "search space change"),
        ),
        (
            "bruteforce-then-generalize",
            "Solve small instances exhaustively first, then infer the general pattern or exploit.",
            ("small cases", "enumeration", "pattern extraction"),
        ),
        (
            "toolchain-specialist",
            "Lean into external tooling already available in the environment: debuggers, solvers, profilers, disassemblers, linters, or test runners.",
            ("tool leverage", "automation", "repeatability"),
        ),
        (
            "simplify-and-isolate",
            "Strip the problem to the smallest reproducible core. Remove irrelevant context aggressively before solving.",
            ("context deletion", "minimal core", "stale-assumption avoidance"),
        ),
    )

    def seeds(self, *, depth: int, parent: BeamCandidate | None, count: int) -> tuple[ThoughtSeed, ...]:
        offset = depth * count
        selected = []
        for index in range(count):
            label, instruction, targets = self._templates[(offset + index) % len(self._templates)]
            parent_hint = ""
            if parent:
                parent_hint = (
                    f"\nContinue from parent `{parent.id}`, but deliberately avoid repeating its exact tactic "
                    "unless the evidence strongly supports it."
                )
            selected.append(
                ThoughtSeed(
                    id=f"seed_{depth}_{index}_{uuid4().hex[:6]}",
                    label=label,
                    instruction=instruction + parent_hint,
                    novelty_targets=targets,
                )
            )
        return tuple(selected)


class BeamSearchOrchestrator:
    def __init__(
        self,
        runtime: AgentRuntime,
        store: ContextStore | None = None,
        diversity: DiversityDirector | None = None,
        config: SearchConfig | None = None,
        role_command: str | None = None,
    ):
        self.runtime = runtime
        self.store = store or ContextStore()
        self.diversity = diversity or DiversityDirector()
        self.config = config or SearchConfig()
        self.role_command = role_command

    def run(
        self,
        *,
        goal: str,
        command: str,
        cwd: str,
        seed_files: dict[str, str | bytes] | None = None,
        metadata: dict[str, Any] | None = None,
        resume_run: RunRecord | None = None,
        resume_message: str | None = None,
    ) -> SearchResult:
        if resume_run:
            run = resume_run
            previous_candidates = list(self.store.list_candidates(run))
            all_candidates = previous_candidates[:]
            frontier = self._frontier_from_candidates(previous_candidates)
            start_depth = (max((candidate.depth for candidate in previous_candidates), default=-1) + 1)
            effective_goal = self._resume_goal(run.goal, resume_message)
            self.store.append_event(
                run,
                "run.resumed",
                {
                    "goal": run.goal,
                    "resume_message": resume_message or "",
                    "metadata": metadata or {},
                    "start_depth": start_depth,
                    "frontier": [candidate.id for candidate in frontier],
                },
            )
            if resume_message:
                self.store.append_event(
                    run,
                    "conversation.message",
                    {
                        "role": "user",
                        "content": resume_message,
                        "start_depth": start_depth,
                    },
                )
        else:
            run = self.store.create_run(goal, metadata=metadata)
            frontier = ()
            all_candidates = []
            start_depth = 0
            effective_goal = goal

        for depth in range(start_depth, start_depth + self.config.max_depth):
            round_candidates, frontier = self._run_depth(
                run=run,
                goal=effective_goal,
                command=command,
                cwd=cwd,
                depth=depth,
                frontier=frontier,
                seed_files=seed_files or {},
            )
            if not round_candidates:
                break
            all_candidates.extend(round_candidates)
            if self.config.stop_on_solved and any(candidate.status == "solved" for candidate in frontier):
                break

        crow_nudges: list[CrowVerdict] = []
        next_depth = start_depth + self.config.max_depth
        if self.config.crow_enabled:
            for nudge_index in range(max(0, self.config.crow_max_nudges)):
                best = max(all_candidates, key=lambda candidate: candidate.score, default=None)
                if best and best.status == "solved":
                    break
                verdict = self._crow_verdict(
                    run=run,
                    goal=effective_goal,
                    candidates=tuple(all_candidates),
                    frontier=frontier,
                    nudge_index=nudge_index,
                    command=self.role_command or command,
                    cwd=cwd,
                )
                crow_nudges.append(verdict)
                self.store.append_event(run, "crow.verdict", asdict(verdict))
                if not verdict.continue_search:
                    break
                for _ in range(verdict.force_depths):
                    round_candidates, frontier = self._run_depth(
                        run=run,
                        goal=effective_goal,
                        command=command,
                        cwd=cwd,
                        depth=next_depth,
                        frontier=frontier,
                        seed_files=seed_files or {},
                    )
                    next_depth += 1
                    if not round_candidates:
                        break
                    all_candidates.extend(round_candidates)
                    if self.config.stop_on_solved and any(candidate.status == "solved" for candidate in frontier):
                        break

        best = max(all_candidates, key=lambda candidate: candidate.score, default=None)
        result = SearchResult(
            run=run,
            best_candidate=best,
            frontier=frontier,
            candidates=tuple(all_candidates),
            crow_nudges=tuple(crow_nudges),
        )
        self.store.write_json(Path(run.root_dir) / "result.json", result.as_dict())
        return result

    def _resume_goal(self, original_goal: str, resume_message: str | None) -> str:
        if not resume_message:
            return original_goal
        return "\n".join(
            [
                "Original user instruction:",
                original_goal,
                "",
                "Latest resume message from the user:",
                resume_message.strip(),
                "",
                "Continue the existing run. Use the stored frontier, role memory, prior events, and candidate contexts as",
                "the conversation history. Treat the latest resume message as the current user turn, not as a replacement",
                "for the original instruction.",
            ]
        ).strip()

    def _run_depth(
        self,
        *,
        run: RunRecord,
        goal: str,
        command: str,
        cwd: str,
        depth: int,
        frontier: tuple[BeamCandidate, ...],
        seed_files: dict[str, str | bytes],
    ) -> tuple[list[BeamCandidate], tuple[BeamCandidate, ...]]:
        parents = frontier[: self.config.beam_width] or (None,)
        scheduled = []
        for parent in parents:
            manager_decision = self._manager_decision(
                run=run,
                goal=goal,
                parent=parent,
                depth=depth,
                frontier=frontier,
                command=self.role_command or command,
                cwd=cwd,
            )
            if manager_decision.stop:
                self.store.append_event(
                    run,
                    "manager.stop",
                    {
                        "depth": depth,
                        "parent": parent.id if parent else None,
                        "rationale": manager_decision.rationale,
                    },
                )
                continue
            diversity_plan = self._diversity_plan(
                run=run,
                goal=goal,
                parent=parent,
                manager_decision=manager_decision,
                depth=depth,
                command=self.role_command or command,
                cwd=cwd,
            )
            for seed in diversity_plan.seeds[: self.config.branch_factor]:
                scheduled.append((parent, seed, manager_decision))
        if not scheduled:
            return [], frontier

        round_candidates = self._run_round(
            run=run,
            goal=goal,
            command=command,
            cwd=cwd,
            depth=depth,
            scheduled=scheduled,
            seed_files=seed_files,
        )
        next_frontier = tuple(
            sorted(round_candidates, key=lambda candidate: candidate.score, reverse=True)[: self.config.beam_width]
        )
        self.store.append_event(
            run,
            "frontier.pruned",
            {
                "depth": depth,
                "kept": [candidate.id for candidate in next_frontier],
                "dropped": [candidate.id for candidate in round_candidates if candidate not in next_frontier],
            },
        )
        return round_candidates, next_frontier

    def _frontier_from_candidates(self, candidates: list[BeamCandidate]) -> tuple[BeamCandidate, ...]:
        if not candidates:
            return ()
        deepest = max(candidate.depth for candidate in candidates)
        deepest_candidates = [candidate for candidate in candidates if candidate.depth == deepest]
        return tuple(
            sorted(deepest_candidates, key=lambda candidate: candidate.score, reverse=True)[: self.config.beam_width]
        )

    def _run_round(
        self,
        *,
        run: RunRecord,
        goal: str,
        command: str,
        cwd: str,
        depth: int,
        scheduled: list[tuple[BeamCandidate | None, ThoughtSeed, ManagerDecision]],
        seed_files: dict[str, str | bytes],
    ) -> list[BeamCandidate]:
        if self.config.max_workers <= 1 or len(scheduled) <= 1:
            return [
                self._run_candidate(
                    run,
                    goal,
                    command,
                    cwd,
                    depth,
                    index,
                    parent,
                    seed,
                    manager_decision,
                    seed_files,
                )
                for index, (parent, seed, manager_decision) in enumerate(scheduled)
            ]

        results: list[BeamCandidate | None] = [None] * len(scheduled)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_candidate,
                    run,
                    goal,
                    command,
                    cwd,
                    depth,
                    index,
                    parent,
                    seed,
                    manager_decision,
                    seed_files,
                ): index
                for index, (parent, seed, manager_decision) in enumerate(scheduled)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return [candidate for candidate in results if candidate is not None]

    def _run_candidate(
        self,
        run: RunRecord,
        goal: str,
        command: str,
        cwd: str,
        depth: int,
        index: int,
        parent: BeamCandidate | None,
        seed: ThoughtSeed,
        manager_decision: ManagerDecision,
        seed_files: dict[str, str | bytes],
    ) -> BeamCandidate:
        candidate_id = f"d{depth}_w{index}_{uuid4().hex[:8]}"
        manager_context = build_manager_context(goal, parent, self.config.context_policy)
        prompt = build_worker_prompt(
            goal=goal,
            manager_context=manager_context,
            manager_decision=manager_decision,
            seed=seed,
            depth=depth,
            candidate_id=candidate_id,
        )
        prompt = self._inject_helm(run, role="worker", name=candidate_id, prompt=prompt)
        prompt_path = self.store.write_prompt(run, "workers", candidate_id, prompt)
        self.store.append_event(
            run,
            "worker.started",
            {
                "candidate": candidate_id,
                "depth": depth,
                "index": index,
                "parent": parent.id if parent else None,
                "seed": asdict(seed),
                "prompt_path": str(prompt_path),
            },
        )
        result = self.runtime.run_agent(
            AgentLaunchSpec(
                name=candidate_id,
                role="worker",
                command=command,
                prompt=prompt,
                cwd=cwd,
                seed_files=seed_files,
                model=self.config.model,
                effort=self.config.effort,
            )
        )
        response_text = result.stdout if result.stdout.strip() else result.stderr
        stored_response = (
            response_text
            if self.config.context_policy.keep_raw_outputs
            else truncate(response_text, self.config.context_policy.max_worker_output_chars)
        )
        response_path = self.store.write_response(run, "workers", candidate_id, stored_response)
        report = parse_candidate_report(response_text, exit_code=result.exit_code)
        next_context = truncate(report.next_context, self.config.context_policy.max_worker_output_chars)
        candidate = BeamCandidate(
            id=candidate_id,
            parent_id=parent.id if parent else None,
            depth=depth,
            seed=seed,
            score=report.score,
            status=report.status,
            summary=report.summary,
            next_context=next_context,
            prompt_path=str(prompt_path),
            response_path=str(response_path),
            artifacts=report.artifacts,
        )
        self.store.write_candidate(run, candidate)
        self._record_invocation(run, candidate, result)
        return candidate

    def _manager_decision(
        self,
        *,
        run: RunRecord,
        goal: str,
        parent: BeamCandidate | None,
        depth: int,
        frontier: tuple[BeamCandidate, ...],
        command: str,
        cwd: str,
    ) -> ManagerDecision:
        fallback_context = build_manager_context(goal, parent, self.config.context_policy)
        if not self.config.agentic_roles:
            return ManagerDecision(
                directive="Continue this branch with a clean worker context.",
                context_to_keep=fallback_context,
                rationale="static manager mode",
            )

        state = self.store.read_role_state(run, "manager")
        prompt = build_manager_prompt(
            goal=goal,
            parent=parent,
            depth=depth,
            frontier=frontier,
            role_state=state,
            policy=self.config.context_policy,
        )
        name = f"manager_d{depth}_{parent.id if parent else 'root'}"
        prompt = self._inject_helm(run, role="manager", name=name, prompt=prompt)
        prompt_path = self.store.write_prompt(run, "manager", name, prompt)
        self.store.append_event(
            run,
            "manager.started",
            {
                "name": name,
                "depth": depth,
                "parent": parent.id if parent else None,
                "prompt_path": str(prompt_path),
            },
        )
        result = self.runtime.run_agent(
            AgentLaunchSpec(
                name=name,
                role="manager",
                command=command,
                prompt=prompt,
                cwd=cwd,
                session_id=state.session_id,
                reuse_session=self.config.reuse_role_sessions,
                model=self.config.model,
                effort=self.config.effort,
            )
        )
        response_text = result.stdout if result.stdout.strip() else result.stderr
        response_path = self.store.write_response(run, "manager", name, response_text)
        decision = parse_manager_decision(response_text, fallback_context=fallback_context)
        self.store.write_role_state(
            run,
            RoleState(
                role="manager",
                session_id=result.session_id if self.config.reuse_role_sessions else None,
                turns=state.turns + 1,
                compact_context=decision.context_to_keep,
            ),
        )
        self.store.append_event(
            run,
            "manager.decision",
            {
                "name": name,
                "depth": depth,
                "parent": parent.id if parent else None,
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "decision": asdict(decision),
                "invocation": {
                    "exit_code": result.exit_code,
                    "sandbox_id": result.sandbox_id,
                    "command": result.command,
                    "session_id": result.session_id,
                },
            },
        )
        return decision

    def _diversity_plan(
        self,
        *,
        run: RunRecord,
        goal: str,
        parent: BeamCandidate | None,
        manager_decision: ManagerDecision,
        depth: int,
        command: str,
        cwd: str,
    ) -> DiversityPlan:
        fallback = self.diversity.seeds(depth=depth, parent=parent, count=self.config.branch_factor)
        if not self.config.agentic_roles:
            return DiversityPlan(seeds=fallback, rationale="static diversity mode")

        state = self.store.read_role_state(run, "diversity")
        prompt = build_diversity_prompt(
            goal=goal,
            parent=parent,
            manager_decision=manager_decision,
            depth=depth,
            count=self.config.branch_factor,
            existing_labels=tuple(seed.label for seed in fallback),
            role_state=state,
        )
        name = f"diversity_d{depth}_{parent.id if parent else 'root'}"
        prompt = self._inject_helm(run, role="diversity", name=name, prompt=prompt)
        prompt_path = self.store.write_prompt(run, "diversity", name, prompt)
        self.store.append_event(
            run,
            "diversity.started",
            {
                "name": name,
                "depth": depth,
                "parent": parent.id if parent else None,
                "prompt_path": str(prompt_path),
            },
        )
        result = self.runtime.run_agent(
            AgentLaunchSpec(
                name=name,
                role="diversity",
                command=command,
                prompt=prompt,
                cwd=cwd,
                session_id=state.session_id,
                reuse_session=self.config.reuse_role_sessions,
                model=self.config.model,
                effort=self.config.effort,
            )
        )
        response_text = result.stdout if result.stdout.strip() else result.stderr
        response_path = self.store.write_response(run, "diversity", name, response_text)
        plan = parse_diversity_plan(response_text, fallback=fallback)
        self.store.write_role_state(
            run,
            RoleState(
                role="diversity",
                session_id=result.session_id if self.config.reuse_role_sessions else None,
                turns=state.turns + 1,
                compact_context=plan.rationale,
            ),
        )
        self.store.append_event(
            run,
            "diversity.plan",
            {
                "name": name,
                "depth": depth,
                "parent": parent.id if parent else None,
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "plan": asdict(plan),
                "invocation": {
                    "exit_code": result.exit_code,
                    "sandbox_id": result.sandbox_id,
                    "command": result.command,
                    "session_id": result.session_id,
                },
            },
        )
        return plan

    def _crow_verdict(
        self,
        *,
        run: RunRecord,
        goal: str,
        candidates: tuple[BeamCandidate, ...],
        frontier: tuple[BeamCandidate, ...],
        nudge_index: int,
        command: str,
        cwd: str,
    ) -> CrowVerdict:
        best = max(candidates, key=lambda candidate: candidate.score, default=None)
        solved = bool(best and best.status == "solved")
        prompt = build_crow_prompt(
            goal=goal,
            candidates=candidates,
            frontier=frontier,
            events=self.store.list_events(run, limit=self.config.crow_event_limit),
            nudge_index=nudge_index,
        )
        name = f"crow_{nudge_index}"
        prompt = self._inject_helm(run, role="crow", name=name, prompt=prompt)
        prompt_path = self.store.write_prompt(run, "crow", name, prompt)
        self.store.append_event(
            run,
            "crow.started",
            {
                "name": name,
                "nudge_index": nudge_index,
                "prompt_path": str(prompt_path),
            },
        )
        result = self.runtime.run_agent(
            AgentLaunchSpec(
                name=name,
                role="crow",
                command=command,
                prompt=prompt,
                cwd=cwd,
                reuse_session=False,
                model=self.config.model,
                effort=self.config.effort,
            )
        )
        response_text = result.stdout if result.stdout.strip() else result.stderr
        response_path = self.store.write_response(run, "crow", name, response_text)
        verdict = parse_crow_verdict(response_text, solved=solved)
        self.store.append_event(
            run,
            "crow.nudge",
            {
                "name": name,
                "nudge_index": nudge_index,
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "verdict": asdict(verdict),
                "invocation": {
                    "exit_code": result.exit_code,
                    "sandbox_id": result.sandbox_id,
                    "command": result.command,
                    "session_id": result.session_id,
                },
            },
        )
        return verdict

    def _inject_helm(self, run: RunRecord, *, role: str, name: str, prompt: str) -> str:
        helm_context = render_helm_context(self.config.helm_directives, role)
        if not helm_context:
            return prompt
        directives = applicable_helm_directives(self.config.helm_directives, role)
        self.store.append_event(
            run,
            "helm.injected",
            {
                "role": role,
                "name": name,
                "directives": [asdict(directive) for directive in directives],
            },
        )
        return f"{helm_context}\n\n---\n\n{prompt}"

    def _record_invocation(self, run: RunRecord, candidate: BeamCandidate, result: AgentResult) -> None:
        self.store.append_event(
            run,
            "worker.invocation",
            {
                "candidate": candidate.id,
                "exit_code": result.exit_code,
                "sandbox_id": result.sandbox_id,
                "command": result.command,
            },
        )
