from __future__ import annotations

import os
import shlex
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol

from .cube import ENV_FILE, read_cube_env


class CommandResultLike(Protocol):
    stdout: str
    stderr: str
    exit_code: int


class SandboxLike(Protocol):
    sandbox_id: str
    commands: Any
    files: Any

    def kill(self) -> None: ...


class AgentRuntime(Protocol):
    def run_agent(self, spec: "AgentLaunchSpec") -> "AgentResult": ...


@dataclass(frozen=True)
class LocalCommandResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class NetworkPolicy:
    allow_internet_access: bool = False
    allow_out: tuple[str, ...] = ()
    deny_out: tuple[str, ...] = ()

    def create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "allow_internet_access": self.allow_internet_access,
        }
        network: dict[str, list[str]] = {}
        if self.allow_out:
            network["allow_out"] = list(self.allow_out)
        if self.deny_out:
            network["deny_out"] = list(self.deny_out)
        if network:
            kwargs["network"] = network
        return kwargs


@dataclass(frozen=True)
class CubeSandboxConfig:
    template_id: str
    timeout: int = 600
    network: NetworkPolicy = field(default_factory=NetworkPolicy)

    @classmethod
    def from_env(cls) -> "CubeSandboxConfig":
        saved_env = read_cube_env(Path.cwd() / ENV_FILE)
        for key, value in saved_env.items():
            os.environ.setdefault(key, value)

        template_id = os.environ.get("CUBE_TEMPLATE_ID")
        if not template_id:
            raise RuntimeError(
                "CUBE_TEMPLATE_ID is required to create CubeSandbox agents. "
                "Run `pawahara-harness cube up` first, or set CubeSandbox environment variables manually."
            )
        timeout = int(os.environ.get("CUBE_SANDBOX_TIMEOUT", "600"))
        allow_internet = os.environ.get("CUBE_ALLOW_INTERNET", "0") in {"1", "true", "yes"}
        return cls(
            template_id=template_id,
            timeout=timeout,
            network=NetworkPolicy(allow_internet_access=allow_internet),
        )


@dataclass(frozen=True)
class AgentLaunchSpec:
    name: str
    role: str
    command: str
    prompt: str = ""
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "/workspace"
    bootstrap_commands: tuple[str, ...] = ()
    seed_files: dict[str, str | bytes] = field(default_factory=dict)
    timeout: int | None = None
    keep_alive: bool = False


@dataclass(frozen=True)
class AgentResult:
    name: str
    role: str
    sandbox_id: str
    command: str
    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class CubeSandboxRuntime:
    def __init__(self, config: CubeSandboxConfig):
        self.config = config

    def create_sandbox(self) -> SandboxLike:
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "e2b-code-interpreter is required. Install with `pip install -e .`."
            ) from exc

        return Sandbox.create(
            template=self.config.template_id,
            timeout=self.config.timeout,
            **self.config.network.create_kwargs(),
        )

    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        sandbox = self.create_sandbox()
        try:
            return self.run_agent_in_sandbox(sandbox, spec)
        finally:
            if not spec.keep_alive:
                cleanup_sandbox(sandbox)

    def run_agent_in_sandbox(self, sandbox: SandboxLike, spec: AgentLaunchSpec) -> AgentResult:
        self._seed_files(sandbox, spec.seed_files)
        for command in spec.bootstrap_commands:
            checked = self._run_shell(sandbox, command, cwd=spec.cwd, env=spec.env, timeout=spec.timeout)
            if checked.exit_code != 0:
                return AgentResult(
                    name=spec.name,
                    role=spec.role,
                    sandbox_id=sandbox.sandbox_id,
                    command=command,
                    stdout=getattr(checked, "stdout", ""),
                    stderr=getattr(checked, "stderr", ""),
                    exit_code=getattr(checked, "exit_code", 1),
                )

        command = build_agent_shell_command(spec.command, prompt=spec.prompt, cwd=spec.cwd, env=spec.env)
        result = self._run_raw(sandbox, command, timeout=spec.timeout)
        return AgentResult(
            name=spec.name,
            role=spec.role,
            sandbox_id=sandbox.sandbox_id,
            command=command,
            stdout=getattr(result, "stdout", ""),
            stderr=getattr(result, "stderr", ""),
            exit_code=getattr(result, "exit_code", 1),
        )

    def _seed_files(self, sandbox: SandboxLike, files: dict[str, str | bytes]) -> None:
        for path, content in files.items():
            parent = str(PurePosixPath(path).parent)
            if parent and parent != ".":
                sandbox.commands.run(f"mkdir -p {shlex.quote(parent)}")
            sandbox.files.write(path, content)

    def _run_shell(
        self,
        sandbox: SandboxLike,
        command: str,
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int | None,
    ) -> CommandResultLike:
        return self._run_raw(sandbox, build_shell_command(command, cwd=cwd, env=env), timeout=timeout)

    def _run_raw(self, sandbox: SandboxLike, command: str, *, timeout: int | None) -> CommandResultLike:
        if timeout is None:
            return sandbox.commands.run(command)
        return sandbox.commands.run(command, timeout=timeout)


class LocalCodexRuntime:
    """Runs the agent command on the host and lets Codex provide its own sandbox."""

    def run_agent(self, spec: AgentLaunchSpec) -> AgentResult:
        for path, content in spec.seed_files.items():
            target = Path(path)
            if not target.is_absolute():
                target = Path(spec.cwd) / target
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")

        for command in spec.bootstrap_commands:
            checked = self._run_shell(command, cwd=spec.cwd, env=spec.env, timeout=spec.timeout)
            if checked.exit_code != 0:
                return AgentResult(
                    name=spec.name,
                    role=spec.role,
                    sandbox_id="codex-local",
                    command=command,
                    stdout=checked.stdout,
                    stderr=checked.stderr,
                    exit_code=checked.exit_code,
                )

        command = build_agent_shell_command(spec.command, prompt=spec.prompt, cwd=spec.cwd, env=spec.env)
        result = self._run_raw(command, timeout=spec.timeout)
        return AgentResult(
            name=spec.name,
            role=spec.role,
            sandbox_id="codex-local",
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    def _run_shell(
        self,
        command: str,
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int | None,
    ) -> LocalCommandResult:
        return self._run_raw(build_shell_command(command, cwd=cwd, env=env), timeout=timeout)

    def _run_raw(self, command: str, *, timeout: int | None) -> LocalCommandResult:
        completed = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return LocalCommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )


class AgentSupervisor:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run_main(self, goal: str, command: str, **kwargs: Any) -> AgentResult:
        return self.runtime.run_agent(
            AgentLaunchSpec(name="main", role="main", command=command, prompt=goal, **kwargs)
        )

    def run_subagent(self, parent: AgentResult, task: str, command: str, name: str, **kwargs: Any) -> AgentResult:
        prompt = textwrap.dedent(
            f"""
            Parent agent: {parent.name}
            Parent sandbox: {parent.sandbox_id}

            Task:
            {task}
            """
        ).strip()
        return self.runtime.run_agent(
            AgentLaunchSpec(name=name, role="subagent", command=command, prompt=prompt, **kwargs)
        )

    def run_subagents(
        self,
        parent: AgentResult,
        tasks: Iterable[str],
        command: str,
        *,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> list[AgentResult]:
        task_list = list(tasks)
        results: list[AgentResult | None] = [None] * len(task_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_subagent,
                    parent,
                    task,
                    command,
                    f"subagent-{index + 1}",
                    **kwargs,
                ): index
                for index, task in enumerate(task_list)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result for result in results if result is not None]


def build_agent_shell_command(command: str, *, prompt: str, cwd: str, env: dict[str, str]) -> str:
    prompt_path = "/tmp/pawahara_agent_prompt.txt"
    parts = [write_file_command(prompt_path, prompt), build_shell_command(f"{command} < {shlex.quote(prompt_path)}", cwd=cwd, env=env)]
    return "\n".join(parts)


def build_shell_command(command: str, *, cwd: str, env: dict[str, str]) -> str:
    exports = "; ".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    setup = f"mkdir -p {shlex.quote(cwd)} && cd {shlex.quote(cwd)}"
    if exports:
        return f"{setup} && {exports}; {command}"
    return f"{setup} && {command}"


def write_file_command(path: str, content: str) -> str:
    delimiter = "__PAWAHARA_PROMPT_EOF__"
    while delimiter in content:
        delimiter = f"_{delimiter}_"
    return f"cat > {shlex.quote(path)} <<'{delimiter}'\n{content}\n{delimiter}"


def cleanup_sandbox(sandbox: SandboxLike) -> None:
    for method_name in ("kill", "close"):
        method = getattr(sandbox, method_name, None)
        if callable(method):
            method()
            return
