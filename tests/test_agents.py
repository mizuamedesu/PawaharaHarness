from __future__ import annotations

from dataclasses import dataclass

from pawahara_harness.agents import (
    AgentLaunchSpec,
    AgentResult,
    AgentSupervisor,
    CubeSandboxConfig,
    CubeSandboxRuntime,
    LocalCodexRuntime,
    build_agent_shell_command,
    write_file_command,
)


@dataclass
class FakeCommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, command: str, timeout: int | None = None) -> FakeCommandResult:
        self.calls.append(command)
        return FakeCommandResult(stdout=f"ran: {command}", exit_code=0)


class FakeFiles:
    def __init__(self) -> None:
        self.writes: dict[str, str | bytes] = {}

    def write(self, path: str, content: str | bytes) -> None:
        self.writes[path] = content


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx_fake") -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands()
        self.files = FakeFiles()
        self.killed = False

    def kill(self) -> None:
        self.killed = True


class FakeRuntime(CubeSandboxRuntime):
    def __init__(self) -> None:
        super().__init__(CubeSandboxConfig(template_id="tpl_fake"))
        self.sandboxes: list[FakeSandbox] = []

    def create_sandbox(self) -> FakeSandbox:
        sandbox = FakeSandbox(f"sbx_{len(self.sandboxes) + 1}")
        self.sandboxes.append(sandbox)
        return sandbox


def test_write_file_command_uses_stable_heredoc() -> None:
    command = write_file_command("/tmp/prompt.txt", "hello\nworld")
    assert "cat > /tmp/prompt.txt <<'__PAWAHARA_PROMPT_EOF__'" in command
    assert "hello\nworld" in command


def test_agent_shell_command_writes_prompt_and_runs_inside_cwd() -> None:
    command = build_agent_shell_command("codex exec", prompt="do work", cwd="/workspace", env={"A": "b c"})
    assert "cat > /tmp/pawahara_agent_" in command
    assert "mkdir -p /workspace && cd /workspace && export A='b c'; codex exec < /tmp/pawahara_agent_" in command


def test_runtime_runs_agent_and_cleans_up_sandbox() -> None:
    runtime = FakeRuntime()
    result = runtime.run_agent(
        AgentLaunchSpec(
            name="main",
            role="main",
            command="agent",
            prompt="hello",
            seed_files={"/workspace/input.txt": "seed"},
        )
    )
    sandbox = runtime.sandboxes[0]
    assert result.ok
    assert result.sandbox_id == "sbx_1"
    assert sandbox.files.writes["/workspace/input.txt"] == "seed"
    assert sandbox.commands.calls[0] == "mkdir -p /workspace"
    assert sandbox.commands.calls
    assert sandbox.killed


def test_local_codex_runtime_runs_command_on_host(tmp_path) -> None:
    runtime = LocalCodexRuntime()
    result = runtime.run_agent(
        AgentLaunchSpec(
            name="main",
            role="main",
            command="cat",
            prompt="hello",
            cwd=str(tmp_path),
        )
    )
    assert result.ok
    assert result.sandbox_id == "codex-local"
    assert result.stdout == "hello\n"


def test_supervisor_launches_subagents() -> None:
    runtime = FakeRuntime()
    supervisor = AgentSupervisor(runtime)
    parent = AgentResult(
        name="main",
        role="main",
        sandbox_id="sbx_parent",
        command="agent",
        stdout="",
        stderr="",
        exit_code=0,
    )
    results = supervisor.run_subagents(parent, ["task a", "task b"], "agent", max_workers=2)
    assert [result.name for result in results] == ["subagent-1", "subagent-2"]
    assert len(runtime.sandboxes) == 2
