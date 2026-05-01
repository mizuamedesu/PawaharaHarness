from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_CODEX_COMMAND = "codex --ask-for-approval never exec --skip-git-repo-check --sandbox workspace-write"
DEFAULT_CUBE_TEMPLATE_IMAGE = "ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pawahara-harness")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run = subparsers.add_parser("run", help="Run a main agent.")
    run.add_argument("--goal", required=True)
    run.add_argument("--command", default=DEFAULT_CODEX_COMMAND, help="Command to execute for the agent.")
    run.add_argument("--backend", choices=["codex", "codex-sdk", "cube"], default="codex")
    run.add_argument("--use-cube", action="store_const", const="cube", dest="backend")
    run.add_argument("--cwd")
    run.add_argument("--model")
    run.add_argument("--effort")
    run.add_argument("--helm", action="append", default=[], help="Force-inject Helm steering text.")
    run.add_argument("--helm-file", action="append", default=[], help="Force-inject Helm steering from a file.")
    run.add_argument("--seed", action="append", default=[], help="Seed file as sandbox_path=local_path.")
    run.add_argument("--keep-alive", action="store_true")
    add_cube_bootstrap_arguments(run, prefix="cube-")

    tui = subparsers.add_parser("tui", help="Open the interactive Pawahara Harness TUI.")
    tui.add_argument("--backend", choices=["codex", "codex-sdk", "cube"], default="codex")
    tui.add_argument("--goal", default="")
    tui.add_argument("--command", default=DEFAULT_CODEX_COMMAND)

    search = subparsers.add_parser("search", help="Run a context-managed diverse beam search.")
    search.add_argument("--goal")
    search.add_argument("--command", default=DEFAULT_CODEX_COMMAND, help="Command to execute for each worker.")
    search.add_argument("--backend", choices=["codex", "codex-sdk", "cube"], default="codex")
    search.add_argument("--use-cube", action="store_const", const="cube", dest="backend")
    search.add_argument("--cwd")
    search.add_argument("--seed", action="append", default=[], help="Seed file as sandbox_path=local_path.")
    search.add_argument("--beam-width", type=int, default=4)
    search.add_argument("--branch-factor", type=int, default=4)
    search.add_argument("--max-depth", type=int, default=2)
    search.add_argument("--max-workers", type=int, default=4)
    search.add_argument("--no-stop-on-solved", action="store_true")
    search.add_argument("--no-agentic-roles", action="store_true")
    search.add_argument("--no-role-sessions", action="store_true")
    search.add_argument("--role-command", help="Command for manager/diversity agents; defaults to --command.")
    search.add_argument("--model")
    search.add_argument("--effort")
    search.add_argument("--max-parent-context-chars", type=int, default=4000)
    search.add_argument("--max-worker-context-chars", type=int, default=12000)
    search.add_argument("--drop-raw-worker-outputs", action="store_true")
    search.add_argument("--runs-dir", default=".pawahara/runs")
    search.add_argument("--resume-run", help="Existing run id or run directory to continue.")
    search.add_argument("--resume-message", help="Additional user message for a resumed run.")
    search.add_argument("--no-crow", action="store_true", help="Disable the independent completion watchdog.")
    search.add_argument("--crow-max-nudges", type=int, default=3)
    search.add_argument("--crow-event-limit", type=int, default=20)
    search.add_argument("--helm", action="append", default=[], help="Force-inject Helm steering text.")
    search.add_argument("--helm-file", action="append", default=[], help="Force-inject Helm steering from a file.")
    add_cube_bootstrap_arguments(search, prefix="cube-")

    cube = subparsers.add_parser("cube", help="Manage local CubeSandbox startup.")
    cube_sub = cube.add_subparsers(dest="cube_command", required=True)

    cube_status = cube_sub.add_parser("status", help="Check CubeSandbox API status.")
    cube_status.add_argument("--api-url", default="http://127.0.0.1:3000")

    cube_up = cube_sub.add_parser("up", help="Start or discover CubeSandbox and create a template.")
    cube_up.add_argument("--mode", choices=["auto", "direct", "dev-vm"], default="auto")
    cube_up.add_argument("--api-url", default="http://127.0.0.1:3000")
    cube_up.add_argument("--dev-vm-api-url", default="http://127.0.0.1:13000")
    cube_up.add_argument("--template-id")
    cube_up.add_argument("--template-image", default=DEFAULT_CUBE_TEMPLATE_IMAGE)
    cube_up.add_argument("--wait-seconds", type=int, default=900)
    cube_up.add_argument("--no-install", action="store_true")
    cube_up.add_argument("--no-create-template", action="store_true")

    args = parser.parse_args(argv)

    if args.command_name == "run":
        from .agents import AgentSupervisor, CubeSandboxConfig, CubeSandboxRuntime

        repo_root = Path.cwd()
        cwd = args.cwd or ("/workspace" if args.backend == "cube" else str(repo_root))
        cube_diagnosis: CubeDiagnosis | None = None
        if args.backend == "cube":
            cube_result = _prepare_cube_backend(args, repo_root)
            if isinstance(cube_result, int):
                return cube_result
            cube_diagnosis = cube_result
            runtime = CubeSandboxRuntime(CubeSandboxConfig.from_env())
        else:
            runtime_or_error = _build_runtime_or_error(args.backend)
            if isinstance(runtime_or_error, int):
                return runtime_or_error
            runtime = runtime_or_error

        supervisor = AgentSupervisor(runtime)
        seed_files = _load_seed_files(args.seed)
        helm_directives = _load_helm_directives(args.helm, args.helm_file)
        goal = _inject_helm_for_role(args.goal, role="main", directives=helm_directives)
        result = supervisor.run_main(
            goal,
            args.command,
            cwd=cwd,
            seed_files=seed_files,
            keep_alive=args.keep_alive,
            model=args.model,
            effort=args.effort,
        )
        payload = {"backend": args.backend, **result.__dict__}
        if cube_diagnosis:
            payload["cube_diagnosis"] = cube_diagnosis.__dict__ | {
                "environment": cube_diagnosis.environment.as_env() if cube_diagnosis.environment else None
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.ok else result.exit_code or 1

    if args.command_name == "tui":
        from .tui import PawaharaTui, TuiSettings

        return PawaharaTui(
            TuiSettings(
                backend=args.backend,
                goal=args.goal,
                command=args.command,
            )
        ).run_loop()

    if args.command_name == "search":
        from dataclasses import asdict

        from .context import ContextPolicy, ContextStore
        from .orchestrator import BeamSearchOrchestrator, SearchConfig

        repo_root = Path.cwd()
        cwd = args.cwd or ("/workspace" if args.backend == "cube" else str(repo_root))
        store = ContextStore(Path(args.runs_dir))
        resume_run = store.load_run(args.resume_run) if args.resume_run else None
        goal = args.goal or (resume_run.goal if resume_run else None)
        if not goal:
            print(
                json.dumps(
                    {"ok": False, "error": "`--goal` is required unless `--resume-run` is provided."},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        cube_diagnosis: CubeDiagnosis | None = None
        if args.backend == "cube":
            cube_result = _prepare_cube_backend(args, repo_root)
            if isinstance(cube_result, int):
                return cube_result
            cube_diagnosis = cube_result
            runtime = CubeSandboxRuntime(CubeSandboxConfig.from_env())
        else:
            runtime_or_error = _build_runtime_or_error(args.backend)
            if isinstance(runtime_or_error, int):
                return runtime_or_error
            runtime = runtime_or_error

        seed_files = _load_seed_files(args.seed)
        helm_directives = _load_helm_directives(args.helm, args.helm_file)
        orchestrator = BeamSearchOrchestrator(
            runtime=runtime,
            store=store,
            config=SearchConfig(
                beam_width=args.beam_width,
                branch_factor=args.branch_factor,
                max_depth=args.max_depth,
                max_workers=args.max_workers,
                stop_on_solved=not args.no_stop_on_solved,
                agentic_roles=not args.no_agentic_roles,
                reuse_role_sessions=not args.no_role_sessions,
                model=args.model,
                effort=args.effort,
                crow_enabled=not args.no_crow,
                crow_max_nudges=args.crow_max_nudges,
                crow_event_limit=args.crow_event_limit,
                helm_directives=helm_directives,
                context_policy=ContextPolicy(
                    max_parent_summary_chars=args.max_parent_context_chars,
                    max_worker_output_chars=args.max_worker_context_chars,
                    keep_raw_outputs=not args.drop_raw_worker_outputs,
                ),
            ),
            role_command=args.role_command,
        )
        result = orchestrator.run(
            goal=goal,
            command=args.command,
            cwd=cwd,
            seed_files=seed_files,
            metadata={
                "backend": args.backend,
                "helm_directives": [asdict(directive) for directive in helm_directives],
                "cube_diagnosis": (
                    cube_diagnosis.__dict__
                    | {"environment": cube_diagnosis.environment.as_env() if cube_diagnosis.environment else None}
                    if cube_diagnosis
                    else None
                ),
            },
            resume_run=resume_run,
            resume_message=args.resume_message,
        )
        print(json.dumps({"backend": args.backend, **result.as_dict()}, ensure_ascii=False, indent=2))
        return 0

    if args.command_name == "cube":
        repo_root = Path.cwd()
        if args.cube_command == "status":
            from .cube import CubeBootstrapOptions, CubeBootstrapper

            bootstrapper = CubeBootstrapper(CubeBootstrapOptions(repo_root=repo_root, api_url=args.api_url))
            env = bootstrapper.status()
            if env is None:
                print(json.dumps({"healthy": False}, ensure_ascii=False, indent=2))
                return 1
            print(json.dumps({"healthy": True, **env.as_env()}, ensure_ascii=False, indent=2))
            return 0

        if args.cube_command == "up":
            from .cube import CubeBootstrapOptions, CubeBootstrapper

            bootstrapper = CubeBootstrapper(
                CubeBootstrapOptions(
                    repo_root=repo_root,
                    api_url=args.api_url,
                    dev_vm_api_url=args.dev_vm_api_url,
                    template_id=args.template_id,
                    template_image=args.template_image,
                    mode=args.mode,
                    wait_seconds=args.wait_seconds,
                    install=not args.no_install,
                    create_template=not args.no_create_template,
                )
            )
            try:
                env = bootstrapper.up()
            except RuntimeError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
                return 1
            print(json.dumps(env.as_env(), ensure_ascii=False, indent=2))
            return 0

    return 2


def _build_runtime_or_error(backend: str) -> AgentRuntime | int:
    try:
        return _build_runtime(backend)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "backend": backend, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


def _build_runtime(backend: str) -> AgentRuntime:
    from .agents import CodexAppServerRuntime, LocalCodexRuntime

    if backend == "codex-sdk":
        runtime = CodexAppServerRuntime()
        runtime.ensure_ready()
        return runtime
    return LocalCodexRuntime()


def add_cube_bootstrap_arguments(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    parser.add_argument(f"--{prefix}mode", choices=["auto", "direct", "dev-vm"], default="auto")
    parser.add_argument(f"--{prefix}api-url", default="http://127.0.0.1:3000")
    parser.add_argument(f"--{prefix}dev-vm-api-url", default="http://127.0.0.1:13000")
    parser.add_argument(f"--{prefix}template-id")
    parser.add_argument(f"--{prefix}template-image", default=DEFAULT_CUBE_TEMPLATE_IMAGE)
    parser.add_argument(f"--{prefix}wait-seconds", type=int, default=900)
    parser.add_argument(f"--{prefix}no-install", action="store_true")
    parser.add_argument(f"--{prefix}no-create-template", action="store_true")


def _prepare_cube_backend(args: argparse.Namespace, repo_root: Path) -> CubeDiagnosis | int:
    from .cube import CubeBootstrapOptions, CubeBootstrapper

    bootstrapper = CubeBootstrapper(
        CubeBootstrapOptions(
            repo_root=repo_root,
            api_url=args.cube_api_url,
            dev_vm_api_url=args.cube_dev_vm_api_url,
            template_id=args.cube_template_id,
            template_image=args.cube_template_image,
            mode=args.cube_mode,
            wait_seconds=args.cube_wait_seconds,
            install=not args.cube_no_install,
            create_template=not args.cube_no_create_template,
        )
    )
    diagnosis = bootstrapper.diagnose()
    if not diagnosis.ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "backend": "cube",
                    "diagnosis": diagnosis.__dict__ | {
                        "environment": diagnosis.environment.as_env() if diagnosis.environment else None
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    try:
        env = bootstrapper.up()
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "backend": "cube",
                    "diagnosis": diagnosis.__dict__ | {
                        "environment": diagnosis.environment.as_env() if diagnosis.environment else None
                    },
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    for key, value in env.as_env().items():
        os.environ[key] = value
    return diagnosis


def _load_seed_files(entries: list[str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--seed must be sandbox_path=local_path, got: {entry}")
        sandbox_path, local_path = entry.split("=", 1)
        files[sandbox_path] = Path(local_path).read_bytes()
    return files


def _load_helm_directives(inline_entries: list[str], file_entries: list[str]) -> tuple[HelmDirective, ...]:
    from .context import HelmDirective

    directives: list[HelmDirective] = []
    for index, entry in enumerate(inline_entries):
        scopes, content = _split_helm_scoped_value(entry)
        if content.strip():
            directives.append(HelmDirective(name=f"inline_{index}", content=content.strip(), scopes=scopes))

    for index, entry in enumerate(file_entries):
        scopes, path_text = _split_helm_scoped_value(entry)
        path = Path(path_text).expanduser()
        content = path.read_text(encoding="utf-8").strip()
        if content:
            directives.append(HelmDirective(name=f"file_{index}:{path.name}", content=content, scopes=scopes))
    return tuple(directives)


def _split_helm_scoped_value(entry: str) -> tuple[tuple[str, ...], str]:
    from .context import HELM_ROLES

    prefix, separator, rest = entry.partition(":")
    if separator:
        scopes = _parse_helm_scopes(prefix)
        if scopes is not None:
            return scopes, rest
    return HELM_ROLES, entry


def _parse_helm_scopes(prefix: str) -> tuple[str, ...] | None:
    from .context import HELM_ROLES

    aliases: dict[str, tuple[str, ...]] = {
        "all": HELM_ROLES,
        "agent": ("main", "subagent", "worker"),
        "agents": ("main", "subagent", "worker"),
        "roles": ("manager", "diversity", "crow"),
        "orchestrator": ("manager", "diversity", "crow"),
        "context": ("manager", "diversity", "worker"),
    }
    expanded: list[str] = []
    parts = [part.strip().lower() for part in prefix.split(",") if part.strip()]
    if not parts:
        return None
    for part in parts:
        if part in aliases:
            expanded.extend(aliases[part])
        elif part in HELM_ROLES:
            expanded.append(part)
        else:
            return None
    deduped = tuple(dict.fromkeys(expanded))
    return deduped or None


def _inject_helm_for_role(prompt: str, *, role: str, directives: tuple[HelmDirective, ...]) -> str:
    from .context import render_helm_context

    helm_context = render_helm_context(directives, role)
    if not helm_context:
        return prompt
    return f"{helm_context}\n\n---\n\n{prompt}"


if __name__ == "__main__":
    raise SystemExit(main())
