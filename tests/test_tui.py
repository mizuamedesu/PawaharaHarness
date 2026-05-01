from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from pawahara_harness.cli import DEFAULT_CODEX_COMMAND
from pawahara_harness.context import ContextStore
from pawahara_harness.tui import (
    ANSI_CURRENT_LINE,
    ANSI_RESET,
    PawaharaTui,
    command_suggestions_for_input,
    complete_command_input,
    format_prompt_line,
    highlight_prompt,
    is_command_line,
    parse_bool,
    parse_escape_action,
    resolve_host_command,
    visible_suggestions,
)


def test_tui_backslash_configures_scalar_settings() -> None:
    ui = PawaharaTui()

    assert ui.handle_backslash(r"\backend codex-sdk")
    assert ui.handle_backslash(r"\use-cube")
    assert ui.handle_backslash(r"\beam-width 8")
    assert ui.handle_backslash(r"\max-depth 3")
    assert ui.handle_backslash(r"\role-command codex exec --json")
    assert ui.handle_backslash(r"\command python3 -c 'import sys; print(sys.stdin.read())'")
    assert ui.handle_backslash(r"\unset role-command")

    assert ui.settings.backend == "cube"
    assert ui.settings.beam_width == 8
    assert ui.settings.max_depth == 3
    assert ui.settings.command == r"python3 -c 'import sys; print(sys.stdin.read())'"
    assert ui.settings.role_command is None


def test_tui_supports_cli_style_boolean_backslash_flags() -> None:
    ui = PawaharaTui()

    assert ui.handle_backslash(r"\no-crow")
    assert ui.handle_backslash(r"\no-agentic-roles")
    assert ui.handle_backslash(r"\drop-raw-worker-outputs")
    assert ui.handle_backslash(r"\cube-no-install")
    assert ui.handle_backslash(r"\toggle keep-alive")

    assert not ui.settings.crow_enabled
    assert not ui.settings.agentic_roles
    assert not ui.settings.keep_raw_outputs
    assert not ui.settings.cube_install
    assert ui.settings.keep_alive


def test_tui_accepts_forward_slash_command_aliases() -> None:
    ui = PawaharaTui()

    assert ui.handle_backslash("/backend codex-sdk")
    assert ui.handle_backslash("/model gpt-5.4")
    assert ui.handle_backslash("/no-crow")

    assert ui.settings.backend == "codex-sdk"
    assert ui.settings.model == "gpt-5.4"
    assert not ui.settings.crow_enabled
    assert is_command_line("/settings")
    assert is_command_line(r"\settings")


def test_tui_suggests_and_completes_slash_commands() -> None:
    root_names = [item.name for item in command_suggestions_for_input("/")]
    assert "run" in root_names
    assert "search" not in root_names
    assert "mode" not in root_names

    filtered_names = [item.name for item in command_suggestions_for_input("/b")]
    assert "backend" in filtered_names
    assert "beam-width" in filtered_names
    assert complete_command_input("/ru", 0) == "/run "
    assert command_suggestions_for_input("/run solve task") == []


def test_tui_suggests_command_arguments_after_command_completion() -> None:
    backend_names = [item.name for item in command_suggestions_for_input("/backend ")]
    assert backend_names == ["codex", "codex-sdk", "cube"]
    assert complete_command_input("/backend ", 1) == "/backend codex-sdk"

    effort_names = [item.name for item in command_suggestions_for_input("/effort ")]
    assert "high" in effort_names
    assert "xhigh" in effort_names


def test_tui_suggests_nested_command_arguments() -> None:
    cube_names = [item.name for item in command_suggestions_for_input("/cube ")]
    assert cube_names == ["status", "up", "use", "settings"]
    assert complete_command_input("/cube ", 2) == "/cube use"

    list_names = [item.name for item in command_suggestions_for_input("/helm ")]
    assert list_names == ["add", "list", "remove", "clear"]
    assert complete_command_input("/helm ", 0) == "/helm add "

    set_names = [item.name for item in command_suggestions_for_input("/set back")]
    assert set_names == ["backend"]
    assert complete_command_input("/set backend", 0) == "/set backend "
    assert complete_command_input("/set backend ", 2) == "/set backend cube"

    toggle_names = [item.name for item in command_suggestions_for_input("/toggle crow")]
    assert "crow" in toggle_names
    assert "crow-enabled" in toggle_names


def test_tui_arrow_escape_sequences_are_parsed() -> None:
    assert parse_escape_action("[A") == "up"
    assert parse_escape_action("[B") == "down"
    assert parse_escape_action("OA") == "up"
    assert parse_escape_action("OB") == "down"
    assert parse_escape_action("[1;2A") == "up"
    assert parse_escape_action("[1;2B") == "down"


def test_tui_suggestion_window_keeps_selection_visible() -> None:
    suggestions = command_suggestions_for_input("/")
    start, visible = visible_suggestions(suggestions, selected=10, max_visible=4)
    visible_names = [item.name for item in visible]

    assert len(visible) == 4
    assert suggestions[10].name in visible_names
    assert start > 0


def test_tui_list_commands_manage_seed_and_helm_entries() -> None:
    ui = PawaharaTui()

    assert ui.handle_backslash(r"\seed /workspace/input.txt=./input.txt")
    assert ui.handle_backslash(r"\helm worker:Prefer differential tests.")
    assert ui.handle_backslash(r"\helm-file crow:./crow_steer.md")
    assert ui.handle_backslash(r"\helm remove 0")

    assert ui.settings.seed_entries == ["/workspace/input.txt=./input.txt"]
    assert ui.settings.helm_entries == []
    assert ui.settings.helm_file_entries == ["crow:./crow_steer.md"]


def test_tui_user_facing_help_and_errors_use_slash_commands() -> None:
    ui = PawaharaTui()
    output = io.StringIO()

    with redirect_stdout(output):
        ui._print_banner()
        ui._print_help()
        ui._print_help("settings")
        ui.handle_backslash("/run")
        ui.handle_backslash("/set")
        ui.handle_backslash("/unset")
        ui.handle_backslash("/toggle")
        ui.handle_backslash("/cube wat")
        ui.handle_backslash("/wat")

    rendered = output.getvalue()
    assert "\\" not in rendered
    assert "/run <instruction>" in rendered
    assert "/search" not in rendered
    assert "/mode " not in rendered
    assert "/set <setting> <value>" in rendered
    assert "/cube status | /cube up | /cube use | /cube settings" in rendered


def test_tui_run_without_value_prompts_for_input() -> None:
    ui = PawaharaTui()
    prompts: list[tuple[str, str]] = []

    def fake_prompt(label: str, placeholder: str, **_kwargs: object) -> None:
        prompts.append((label, placeholder))
        return None

    ui._prompt_for_value = fake_prompt  # type: ignore[method-assign]

    assert ui.handle_backslash("/run")
    assert prompts == [("run", "Type the instruction for the harness.")]
    assert ui.settings.goal == ""


def test_tui_run_with_value_dispatches_harness() -> None:
    ui = PawaharaTui()
    calls: list[str] = []

    def fake_execute_search() -> None:
        calls.append(ui.settings.goal)

    ui._execute_search = fake_execute_search  # type: ignore[method-assign]

    assert ui.handle_backslash("/run solve every task")
    assert calls == ["solve every task"]


def test_tui_resume_suggests_previous_runs_and_dispatches_message(tmp_path: Path) -> None:
    store = ContextStore(tmp_path / "runs")
    run = store.create_run("solve old task")
    ui = PawaharaTui()
    ui.settings.runs_dir = str(store.runs_dir)
    calls: list[tuple[str | None, str | None]] = []

    suggestions = ui._suggestions_for_input("/resume ")
    assert suggestions
    assert suggestions[0].name == run.run_id
    assert complete_command_input("/resume ", 0, suggestions=suggestions) == f"/resume {run.run_id} "

    def fake_execute_search(*, resume_message: str | None = None) -> None:
        calls.append((ui.settings.resume_run, resume_message))

    ui._execute_search = fake_execute_search  # type: ignore[method-assign]

    assert ui.handle_backslash(f"/resume {run.run_id} continue from here")
    assert calls == [(run.run_id, "continue from here")]
    assert ui.settings.goal == "continue from here"


def test_tui_search_command_is_not_an_alias() -> None:
    ui = PawaharaTui()
    output = io.StringIO()

    with redirect_stdout(output):
        assert ui.handle_backslash("/search solve every task")

    assert "Unknown command: /search" in output.getvalue()


def test_tui_resolves_discovered_codex_command() -> None:
    resolved, error = resolve_host_command(
        "codex exec --json",
        which=lambda _name: None,
        discover_codex=lambda: Path("/opt/codex/bin/codex"),
    )

    assert error is None
    assert resolved == "/opt/codex/bin/codex exec --json"


def test_tui_rewrites_legacy_codex_approval_flag() -> None:
    resolved, error = resolve_host_command(
        "codex exec --skip-git-repo-check --sandbox workspace-write --approval-policy never",
        which=lambda _name: None,
        discover_codex=lambda: Path("/opt/codex/bin/codex"),
    )

    assert error is None
    assert resolved == (
        "/opt/codex/bin/codex --ask-for-approval never "
        "exec --skip-git-repo-check --sandbox workspace-write"
    )


def test_default_codex_command_uses_current_cli_approval_flag() -> None:
    assert "--approval-policy" not in DEFAULT_CODEX_COMMAND
    assert DEFAULT_CODEX_COMMAND.startswith("codex --ask-for-approval never exec ")


def test_tui_blocks_missing_host_command_before_fanout() -> None:
    resolved, error = resolve_host_command(
        "codex exec --json",
        which=lambda _name: None,
        discover_codex=lambda: None,
    )

    assert resolved == "codex exec --json"
    assert error is not None
    assert "not found in PATH" in error


def test_tui_interrupt_handler_returns_to_prompt_without_traceback() -> None:
    ui = PawaharaTui()
    output = io.StringIO()

    with redirect_stdout(output):
        ui._handle_interrupt()

    assert ui.status_line == "interrupted"
    assert "Interrupted. Use /exit to quit." in output.getvalue()
    assert "Traceback" not in output.getvalue()


def test_tui_inline_input_prompt_marks_goal_as_required() -> None:
    ui = PawaharaTui()

    prompt = ui._input_prompt_for_text("/run ")
    assert prompt is not None
    assert prompt.label == "run"
    assert prompt.required
    assert prompt.value == ""

    prompt = ui._input_prompt_for_text("/run solve ctf")
    assert prompt is not None
    assert prompt.value == "solve ctf"


def test_tui_highlights_prompt_only() -> None:
    rendered = highlight_prompt("pawahara:codex> ")

    assert rendered.startswith(ANSI_CURRENT_LINE)
    assert rendered == f"{ANSI_CURRENT_LINE}pawahara:codex>{ANSI_RESET} "

    line = format_prompt_line("pawahara:codex> ", "/run hello")
    assert line == f"{ANSI_CURRENT_LINE}pawahara:codex>{ANSI_RESET} /run hello"


def test_parse_bool_accepts_tui_terms() -> None:
    assert parse_bool("on")
    assert parse_bool("enabled")
    assert not parse_bool("off")
    assert not parse_bool("disabled")
