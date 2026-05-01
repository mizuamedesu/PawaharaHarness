from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agents import AgentSupervisor, CubeSandboxConfig, CubeSandboxRuntime, LocalCodexRuntime
from .cube import DEFAULT_TEMPLATE_IMAGE, CubeBootstrapOptions, CubeBootstrapper, CubeDiagnosis


DEFAULT_CODEX_COMMAND = "codex exec --skip-git-repo-check --sandbox workspace-write --approval-policy never"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pawahara-harness")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run = subparsers.add_parser("run", help="Run a main agent.")
    run.add_argument("--goal", required=True)
    run.add_argument("--command", default=DEFAULT_CODEX_COMMAND, help="Command to execute for the agent.")
    run.add_argument("--backend", choices=["codex", "cube"], default="codex")
    run.add_argument("--use-cube", action="store_const", const="cube", dest="backend")
    run.add_argument("--cwd")
    run.add_argument("--seed", action="append", default=[], help="Seed file as sandbox_path=local_path.")
    run.add_argument("--keep-alive", action="store_true")
    add_cube_bootstrap_arguments(run, prefix="cube-")

    cube = subparsers.add_parser("cube", help="Manage local CubeSandbox startup.")
    cube_sub = cube.add_subparsers(dest="cube_command", required=True)

    cube_status = cube_sub.add_parser("status", help="Check CubeSandbox API status.")
    cube_status.add_argument("--api-url", default="http://127.0.0.1:3000")

    cube_up = cube_sub.add_parser("up", help="Start or discover CubeSandbox and create a template.")
    cube_up.add_argument("--mode", choices=["auto", "direct", "dev-vm"], default="auto")
    cube_up.add_argument("--api-url", default="http://127.0.0.1:3000")
    cube_up.add_argument("--dev-vm-api-url", default="http://127.0.0.1:13000")
    cube_up.add_argument("--template-id")
    cube_up.add_argument("--template-image", default=DEFAULT_TEMPLATE_IMAGE)
    cube_up.add_argument("--wait-seconds", type=int, default=900)
    cube_up.add_argument("--no-install", action="store_true")
    cube_up.add_argument("--no-create-template", action="store_true")

    args = parser.parse_args(argv)

    if args.command_name == "run":
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
            runtime = LocalCodexRuntime()

        supervisor = AgentSupervisor(runtime)
        seed_files = _load_seed_files(args.seed)
        result = supervisor.run_main(
            args.goal,
            args.command,
            cwd=cwd,
            seed_files=seed_files,
            keep_alive=args.keep_alive,
        )
        payload = {"backend": args.backend, **result.__dict__}
        if cube_diagnosis:
            payload["cube_diagnosis"] = cube_diagnosis.__dict__ | {
                "environment": cube_diagnosis.environment.as_env() if cube_diagnosis.environment else None
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.ok else result.exit_code or 1

    if args.command_name == "cube":
        repo_root = Path.cwd()
        if args.cube_command == "status":
            bootstrapper = CubeBootstrapper(CubeBootstrapOptions(repo_root=repo_root, api_url=args.api_url))
            env = bootstrapper.status()
            if env is None:
                print(json.dumps({"healthy": False}, ensure_ascii=False, indent=2))
                return 1
            print(json.dumps({"healthy": True, **env.as_env()}, ensure_ascii=False, indent=2))
            return 0

        if args.cube_command == "up":
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


def add_cube_bootstrap_arguments(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    parser.add_argument(f"--{prefix}mode", choices=["auto", "direct", "dev-vm"], default="auto")
    parser.add_argument(f"--{prefix}api-url", default="http://127.0.0.1:3000")
    parser.add_argument(f"--{prefix}dev-vm-api-url", default="http://127.0.0.1:13000")
    parser.add_argument(f"--{prefix}template-id")
    parser.add_argument(f"--{prefix}template-image", default=DEFAULT_TEMPLATE_IMAGE)
    parser.add_argument(f"--{prefix}wait-seconds", type=int, default=900)
    parser.add_argument(f"--{prefix}no-install", action="store_true")
    parser.add_argument(f"--{prefix}no-create-template", action="store_true")


def _prepare_cube_backend(args: argparse.Namespace, repo_root: Path) -> CubeDiagnosis | int:
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


if __name__ == "__main__":
    raise SystemExit(main())
