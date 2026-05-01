from __future__ import annotations

import json
import shlex
import shutil
import sys
import termios
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .agents import AgentRuntime, CubeSandboxConfig, CubeSandboxRuntime
from .cli import (
    DEFAULT_CODEX_COMMAND,
    _build_runtime,
    _load_helm_directives,
    _load_seed_files,
    _prepare_cube_backend,
)
from .context import ContextPolicy, ContextStore
from .cube import DEFAULT_TEMPLATE_IMAGE, CubeDiagnosis
from .orchestrator import BeamSearchOrchestrator, SearchConfig


BoolParse = Literal["bool", "int", "str"]


CORE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("run", "run the orchestrated harness"),
    ("settings", "show current config"),
    ("set", "set a scalar setting"),
    ("unset", "clear model/cwd/resume-run/etc."),
    ("toggle", "flip a boolean setting"),
    ("seed", "manage seed files"),
    ("helm", "manage forced steering text"),
    ("helm-file", "manage forced steering files"),
    ("cube", "CubeSandbox status/start/use"),
    ("last", "print last JSON payload"),
    ("clear", "clear the screen"),
    ("help", "show command help"),
    ("exit", "quit"),
)

LIST_ACTIONS = ("add", "list", "remove", "clear")
CUBE_ACTIONS = ("status", "up", "use", "settings")
BOOLEAN_CHOICES = ("true", "false")
EFFORT_CHOICES = ("low", "medium", "high", "xhigh")
MODEL_SUGGESTIONS = ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.2")
MAX_VISIBLE_SUGGESTIONS = 8
ANSI_CURRENT_LINE = "\033[7m"
ANSI_RESET = "\033[0m"


@dataclass(frozen=True)
class SettingSpec:
    attr: str
    kind: BoolParse
    choices: tuple[str, ...] = ()
    nullable: bool = False
    inverted: bool = False


@dataclass(frozen=True)
class CommandSuggestion:
    name: str
    description: str
    completion_suffix: str = ""


@dataclass(frozen=True)
class SlashInput:
    prefix: str
    command_token: str
    arg_text: str | None


@dataclass(frozen=True)
class ArgumentContext:
    parts: tuple[str, ...]
    current: str
    trailing_space: bool


@dataclass(frozen=True)
class TextPrompt:
    label: str
    placeholder: str
    required: bool = True
    value: str = ""


@dataclass
class TuiSettings:
    goal: str = ""
    command: str = DEFAULT_CODEX_COMMAND
    backend: str = "codex"
    cwd: str | None = None
    model: str | None = None
    effort: str | None = None
    keep_alive: bool = False
    seed_entries: list[str] = field(default_factory=list)
    beam_width: int = 4
    branch_factor: int = 4
    max_depth: int = 2
    max_workers: int = 4
    stop_on_solved: bool = True
    agentic_roles: bool = True
    reuse_role_sessions: bool = True
    role_command: str | None = None
    max_parent_context_chars: int = 4000
    max_worker_context_chars: int = 12000
    keep_raw_outputs: bool = True
    runs_dir: str = ".pawahara/runs"
    resume_run: str | None = None
    crow_enabled: bool = True
    crow_max_nudges: int = 3
    crow_event_limit: int = 20
    helm_entries: list[str] = field(default_factory=list)
    helm_file_entries: list[str] = field(default_factory=list)
    cube_mode: str = "auto"
    cube_api_url: str = "http://127.0.0.1:3000"
    cube_dev_vm_api_url: str = "http://127.0.0.1:13000"
    cube_template_id: str | None = None
    cube_template_image: str = DEFAULT_TEMPLATE_IMAGE
    cube_wait_seconds: int = 900
    cube_install: bool = True
    cube_create_template: bool = True


SETTING_SPECS: dict[str, SettingSpec] = {
    "command": SettingSpec("command", "str"),
    "backend": SettingSpec("backend", "str", choices=("codex", "codex-sdk", "cube")),
    "use_cube": SettingSpec("backend", "str", choices=("cube",)),
    "cwd": SettingSpec("cwd", "str", nullable=True),
    "model": SettingSpec("model", "str", nullable=True),
    "effort": SettingSpec("effort", "str", nullable=True),
    "keep_alive": SettingSpec("keep_alive", "bool"),
    "beam_width": SettingSpec("beam_width", "int"),
    "branch_factor": SettingSpec("branch_factor", "int"),
    "max_depth": SettingSpec("max_depth", "int"),
    "max_workers": SettingSpec("max_workers", "int"),
    "stop_on_solved": SettingSpec("stop_on_solved", "bool"),
    "no_stop_on_solved": SettingSpec("stop_on_solved", "bool", inverted=True),
    "agentic_roles": SettingSpec("agentic_roles", "bool"),
    "no_agentic_roles": SettingSpec("agentic_roles", "bool", inverted=True),
    "reuse_role_sessions": SettingSpec("reuse_role_sessions", "bool"),
    "role_sessions": SettingSpec("reuse_role_sessions", "bool"),
    "no_role_sessions": SettingSpec("reuse_role_sessions", "bool", inverted=True),
    "role_command": SettingSpec("role_command", "str", nullable=True),
    "max_parent_context_chars": SettingSpec("max_parent_context_chars", "int"),
    "max_worker_context_chars": SettingSpec("max_worker_context_chars", "int"),
    "keep_raw_outputs": SettingSpec("keep_raw_outputs", "bool"),
    "drop_raw_worker_outputs": SettingSpec("keep_raw_outputs", "bool", inverted=True),
    "runs_dir": SettingSpec("runs_dir", "str"),
    "resume_run": SettingSpec("resume_run", "str", nullable=True),
    "crow_enabled": SettingSpec("crow_enabled", "bool"),
    "crow": SettingSpec("crow_enabled", "bool"),
    "no_crow": SettingSpec("crow_enabled", "bool", inverted=True),
    "crow_max_nudges": SettingSpec("crow_max_nudges", "int"),
    "crow_event_limit": SettingSpec("crow_event_limit", "int"),
    "cube_mode": SettingSpec("cube_mode", "str", choices=("auto", "direct", "dev-vm")),
    "cube_api_url": SettingSpec("cube_api_url", "str"),
    "cube_dev_vm_api_url": SettingSpec("cube_dev_vm_api_url", "str"),
    "cube_template_id": SettingSpec("cube_template_id", "str", nullable=True),
    "cube_template_image": SettingSpec("cube_template_image", "str"),
    "cube_wait_seconds": SettingSpec("cube_wait_seconds", "int"),
    "cube_install": SettingSpec("cube_install", "bool"),
    "cube_no_install": SettingSpec("cube_install", "bool", inverted=True),
    "cube_create_template": SettingSpec("cube_create_template", "bool"),
    "cube_no_create_template": SettingSpec("cube_create_template", "bool", inverted=True),
}


class PawaharaTui:
    def __init__(self, settings: TuiSettings | None = None):
        self.settings = settings or TuiSettings()
        self.last_payload: dict[str, Any] | None = None
        self.status_line = "idle"
        self._rendered_lines = 0

    def run_loop(self) -> int:
        interactive = self._interactive_input_enabled()
        self._print_banner()
        try:
            while True:
                try:
                    line = self._read_line()
                except EOFError:
                    print()
                    return 0
                except KeyboardInterrupt:
                    self.status_line = "interrupted"
                    self._clear_rendered_input()
                    print("\nInterrupted. Use /exit to quit.")
                    continue

                if line is None:
                    return 0
                stripped = line.strip()
                if not stripped:
                    continue
                self.status_line = f"accepted: {truncate_for_status(stripped)}"
                if is_command_line(stripped):
                    if not self.handle_backslash(stripped):
                        return 0
                    continue

                self.settings.goal = stripped
                self._execute_search()
        finally:
            if interactive:
                self._clear_rendered_input()

    def _interactive_input_enabled(self) -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    def _read_line(self, prompt: TextPrompt | None = None) -> str | None:
        if self._interactive_input_enabled():
            return self._read_line_interactive(prompt)
        return input(self._prompt(prompt))

    def _read_line_interactive(self, prompt: TextPrompt | None = None) -> str | None:
        buffer: list[str] = []
        selected = 0
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            new_settings = termios.tcgetattr(sys.stdin.fileno())
            new_settings[3] &= ~(termios.ECHO | termios.ICANON)
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, new_settings)
            while True:
                text = "".join(buffer)
                suggestions = [] if prompt else command_suggestions_for_input(text)
                if suggestions:
                    selected = min(selected, len(suggestions) - 1)
                else:
                    selected = 0
                self._render_composer(text, selected, prompt)
                char = sys.stdin.read(1)
                if char in {"\r", "\n"}:
                    text = "".join(buffer)
                    if prompt:
                        if prompt.required and not text.strip():
                            self.status_line = f"waiting for {prompt.label}"
                            continue
                        self._clear_rendered_input()
                        print(f"{self._prompt(prompt)}{text}")
                        return text
                    completed = complete_command_input(text, selected)
                    if completed is not None and completed != "".join(buffer):
                        buffer = list(completed)
                        selected = 0
                        continue
                    inline_prompt = self._input_prompt_for_text(text)
                    if inline_prompt and inline_prompt.required and not inline_prompt.value.strip():
                        self.status_line = f"waiting for {inline_prompt.label}"
                        continue
                    self._clear_rendered_input()
                    print(f"{self._prompt()}{text}")
                    return text
                if char == "\x03":
                    raise KeyboardInterrupt
                if char == "\x04":
                    self._clear_rendered_input()
                    return None
                if char in {"\x7f", "\b"}:
                    if buffer:
                        buffer.pop()
                        selected = 0
                    continue
                if char == "\t":
                    completed = complete_command_input("".join(buffer), selected)
                    if completed is not None:
                        buffer = list(completed)
                        selected = 0
                    continue
                if char == "\x1b":
                    action = read_escape_action()
                    if action == "up" and suggestions:
                        selected = (selected - 1) % len(suggestions)
                    elif action == "down" and suggestions:
                        selected = (selected + 1) % len(suggestions)
                    elif action == "escape":
                        if prompt:
                            self._clear_rendered_input()
                            return None
                        if suggestions:
                            buffer.clear()
                        else:
                            return ""
                    continue
                if char == "\x10" and suggestions:
                    selected = (selected - 1) % len(suggestions)
                    continue
                if char == "\x0e" and suggestions:
                    selected = (selected + 1) % len(suggestions)
                    continue
                if char.isprintable():
                    buffer.append(char)
                    selected = 0
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

    def _render_composer(self, text: str, selected: int, prompt: TextPrompt | None = None) -> None:
        suggestions = [] if prompt else command_suggestions_for_input(text)
        input_prompt = prompt or (None if suggestions else self._input_prompt_for_text(text))
        lines = [
            (
            f"backend {self.settings.backend}  depth {self.settings.max_depth}  "
            f"beam {self.settings.beam_width}  status {self.status_line}\n"
            ).rstrip(),
            "─" * 72,
        ]
        if input_prompt:
            lines.extend(
                [
                    f"input required: {input_prompt.label}",
                    f"› {input_prompt.placeholder}",
                    format_prompt_line(self._prompt(prompt) if prompt else self._prompt(), text),
                ]
            )
        else:
            lines.append(format_prompt_line(self._prompt(), text))
        if suggestions:
            group = suggestion_group_for_input(text)
            start, visible = visible_suggestions(suggestions, selected)
            lines.extend(["", group])
            if start:
                lines.append("  ...")
            for offset, suggestion in enumerate(visible):
                index = start + offset
                marker = ">" if index == selected else " "
                name = f"/{suggestion.name}" if group == "commands" else f" {suggestion.name}"
                lines.append(f"{marker} {name:<29} {suggestion.description}")
            if start + len(visible) < len(suggestions):
                lines.append("  ...")
            lines.extend(["", "Up/Down select | Enter choose | Tab complete | Esc clear"])
        elif input_prompt:
            lines.extend(["", "Enter submit input | Esc cancel | Ctrl-D exit"])
        else:
            lines.extend(["", "Type / for commands | Enter submit | Ctrl-D exit"])
        self._clear_rendered_input()
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()
        self._rendered_lines = len(lines)

    def _clear_rendered_input(self) -> None:
        if self._rendered_lines <= 0:
            return
        sys.stdout.write("\r")
        if self._rendered_lines > 1:
            sys.stdout.write(f"\033[{self._rendered_lines - 1}A")
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        self._rendered_lines = 0

    def _input_prompt_for_text(self, text: str) -> TextPrompt | None:
        parsed = parse_slash_input(text)
        if parsed is None or parsed.arg_text is None:
            return None
        command = normalize_key(parsed.command_token)
        arg_text = parsed.arg_text
        context = argument_context(arg_text)
        if command == "run" and not self.settings.goal:
            return TextPrompt("run", "Type the instruction for the harness.", value=arg_text.strip())
        if command == "set":
            if len(context.parts) >= 1:
                key = context.parts[0]
                value = arg_text.split(maxsplit=1)[1] if len(arg_text.split(maxsplit=1)) == 2 else ""
                spec = SETTING_SPECS.get(normalize_key(key))
                if spec and setting_requires_value(spec):
                    return TextPrompt(f"{key.replace('_', '-')} value", f"Type a value for {key}.", value=value)
            return None
        if command in {"seed", "helm", "helm_file"}:
            if context.parts and normalize_key(context.parts[0]) in {"add", "append"}:
                value = arg_text.split(maxsplit=1)[1] if len(arg_text.split(maxsplit=1)) == 2 else ""
                return TextPrompt(command.replace("_", "-"), "Type the value to add.", value=value)
            if context.parts and normalize_key(context.parts[0]) in {"remove", "rm", "delete"}:
                value = arg_text.split(maxsplit=1)[1] if len(arg_text.split(maxsplit=1)) == 2 else ""
                return TextPrompt(command.replace("_", "-"), "Type the index to remove.", value=value)
            return None
        spec = SETTING_SPECS.get(command)
        if spec and setting_requires_value(spec):
            return TextPrompt(command.replace("_", "-"), f"Type a value for {command.replace('_', '-')}.", value=arg_text)
        return None

    def _prompt_for_value(
        self,
        label: str,
        placeholder: str,
        *,
        usage: str | None = None,
        required: bool = True,
    ) -> str | None:
        if not self._interactive_input_enabled():
            if usage:
                print(usage)
            return None
        previous_status = self.status_line
        self.status_line = f"waiting for {label}"
        try:
            value = self._read_line(TextPrompt(label=label, placeholder=placeholder, required=required))
        except EOFError:
            print()
            return None
        finally:
            if self.status_line == f"waiting for {label}":
                self.status_line = previous_status
        if value is None:
            print(f"{label}: cancelled")
            return None
        value = value.strip()
        if required and not value:
            if usage:
                print(usage)
            return None
        return value

    def handle_backslash(self, line: str) -> bool:
        command_line = line[1:].strip() if is_command_line(line.strip()) else line.strip()
        if not command_line:
            self._print_help()
            return True

        raw_command, _, raw_rest = command_line.partition(" ")
        try:
            parts = shlex.split(command_line)
        except ValueError as exc:
            print(f"Could not parse command: {exc}")
            return True
        if not parts:
            return True

        command = normalize_key(raw_command)
        args = parts[1:]

        if command in {"exit", "quit", "q"}:
            return False
        if command in {"help", "h", "?"}:
            self._print_help(args[0] if args else None)
            return True
        if command in {"settings", "config"}:
            self._print_settings()
            return True
        if command == "set":
            self._command_set(raw_rest)
            return True
        if command == "unset":
            self._command_unset(args)
            return True
        if command == "toggle":
            self._command_toggle(args)
            return True
        if command == "seed":
            self._command_list_value("seed", self.settings.seed_entries, args)
            return True
        if command == "helm":
            self._command_list_value("helm", self.settings.helm_entries, args)
            return True
        if command == "helm_file":
            self._command_list_value("helm-file", self.settings.helm_file_entries, args)
            return True
        if command == "run":
            if raw_rest.strip():
                self.settings.goal = raw_rest.strip()
            self._execute_search()
            return True
        if command == "last":
            self._print_last_payload()
            return True
        if command == "clear":
            print("\033c", end="")
            self._print_banner()
            return True
        if command == "cube":
            self._command_cube(args)
            return True
        if command in SETTING_SPECS:
            self._set_value(command, raw_rest.strip() if raw_rest.strip() else None)
            return True

        print(f"Unknown command: /{parts[0]}. Use /help.")
        return True

    def _command_set(self, raw_args: str) -> None:
        key, separator, value = raw_args.strip().partition(" ")
        if not key:
            prompted = self._prompt_for_value(
                "setting",
                "Type a setting name, optionally followed by its value.",
                usage="Usage: /set <setting> <value>",
            )
            if prompted is None:
                return
            key, separator, value = prompted.partition(" ")
        if not separator:
            prompted_value = self._prompt_for_value(
                f"{key} value",
                f"Type a value for {key}.",
                usage="Usage: /set <setting> <value>",
            )
            if prompted_value is None:
                return
            value = prompted_value
        self._set_value(key, value)

    def _command_unset(self, args: list[str]) -> None:
        if len(args) != 1:
            prompted = self._prompt_for_value(
                "setting",
                "Type the nullable setting to clear.",
                usage="Usage: /unset <setting>",
            )
            if prompted is None:
                return
            args = [prompted]
        key = normalize_key(args[0])
        spec = SETTING_SPECS.get(key)
        if not spec:
            print(f"Unknown setting: {args[0]}")
            return
        if not spec.nullable:
            print(f"{args[0]} cannot be unset.")
            return
        setattr(self.settings, spec.attr, None)
        print(f"{spec.attr} = <auto>")

    def _command_toggle(self, args: list[str]) -> None:
        if len(args) != 1:
            prompted = self._prompt_for_value(
                "boolean setting",
                "Type the boolean setting to flip.",
                usage="Usage: /toggle <boolean-setting>",
            )
            if prompted is None:
                return
            args = [prompted]
        key = normalize_key(args[0])
        spec = SETTING_SPECS.get(key)
        if not spec or spec.kind != "bool":
            print(f"Not a boolean setting: {args[0]}")
            return
        current = bool(getattr(self.settings, spec.attr))
        setattr(self.settings, spec.attr, not current)
        print(f"{spec.attr} = {not current}")

    def _command_list_value(self, name: str, values: list[str], args: list[str]) -> None:
        action = normalize_key(args[0]) if args else "list"
        rest = args[1:] if args else []
        if action in {"list", "ls"}:
            self._print_list(name, values)
            return
        if action in {"clear", "reset"}:
            values.clear()
            print(f"{name}: cleared")
            return
        if action in {"remove", "rm", "delete"}:
            if len(rest) != 1:
                prompted = self._prompt_for_value(
                    f"{name} index",
                    f"Type the {name} index to remove.",
                    usage=f"Usage: /{name} remove <index>",
                )
                if prompted is None:
                    return
                rest = [prompted]
            try:
                index = int(rest[0])
                removed = values.pop(index)
            except (ValueError, IndexError):
                print(f"Invalid {name} index: {rest[0]}")
                return
            print(f"{name}: removed {removed}")
            return
        if action in {"add", "append"}:
            value = " ".join(rest)
        else:
            value = " ".join(args)
        if not value.strip():
            prompted = self._prompt_for_value(
                name,
                f"Type the {name} value to add.",
                usage=f"Usage: /{name} <value>",
            )
            if prompted is None:
                return
            value = prompted
        values.append(value.strip())
        print(f"{name}: added {value.strip()}")

    def _command_cube(self, args: list[str]) -> None:
        action = normalize_key(args[0]) if args else "settings"
        if action == "settings":
            self._print_settings(prefix="cube_")
            return
        if action == "use":
            self.settings.backend = "cube"
            print("backend = cube")
            return
        if action in {"status", "up"}:
            repo_root = Path.cwd()
            namespace = self._cube_namespace()
            if action == "status":
                from .cube import CubeBootstrapOptions, CubeBootstrapper

                bootstrapper = CubeBootstrapper(
                    CubeBootstrapOptions(repo_root=repo_root, api_url=self.settings.cube_api_url)
                )
                env = bootstrapper.status()
                payload: dict[str, Any] = {"healthy": env is not None}
                if env:
                    payload.update(env.as_env())
                self.last_payload = payload
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return
            result = _prepare_cube_backend(namespace, repo_root)
            if isinstance(result, int):
                print("CubeSandbox startup failed.")
                return
            payload = result.__dict__ | {
                "environment": result.environment.as_env() if result.environment else None,
            }
            self.last_payload = payload
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        if action in SETTING_SPECS:
            self._set_value(action, " ".join(args[1:]) if len(args) > 1 else None)
            return
        print("Usage: /cube status | /cube up | /cube use | /cube settings")

    def _set_value(self, key: str, raw_value: str | None) -> None:
        normalized = normalize_key(key)
        spec = SETTING_SPECS.get(normalized)
        if not spec:
            print(f"Unknown setting: {key}")
            return
        if (raw_value is None or raw_value == "") and setting_requires_value(spec):
            raw_value = self._prompt_for_value(
                key.replace("_", "-"),
                f"Type a value for {key.replace('_', '-')}.",
                usage=f"Usage: /{key.replace('_', '-')} <value>",
            )
            if raw_value is None:
                return
        try:
            value = parse_setting_value(spec, raw_value)
        except ValueError as exc:
            print(str(exc))
            return
        setattr(self.settings, spec.attr, value)
        print(f"{spec.attr} = {format_value(value)}")

    def _execute_search(self) -> None:
        store = ContextStore(Path(self.settings.runs_dir))
        resume_run = store.load_run(self.settings.resume_run) if self.settings.resume_run else None
        goal = self.settings.goal or (resume_run.goal if resume_run else "")
        if not goal:
            goal = self._prompt_for_value(
                "run",
                "Type the instruction for the harness.",
                usage="Set an instruction first: /run <text> or type a plain prompt.",
            )
            if goal is None:
                return
            self.settings.goal = goal
        if not self._prepare_local_agent_commands():
            return
        self.status_line = "running run"
        print("running run...")
        prepared = self._prepare_runtime()
        if prepared is None:
            self.status_line = "run failed before launch"
            return
        runtime, cube_diagnosis = prepared
        seed_files = _load_seed_files(self.settings.seed_entries)
        helm_directives = _load_helm_directives(self.settings.helm_entries, self.settings.helm_file_entries)
        orchestrator = BeamSearchOrchestrator(
            runtime=runtime,
            store=store,
            config=SearchConfig(
                beam_width=self.settings.beam_width,
                branch_factor=self.settings.branch_factor,
                max_depth=self.settings.max_depth,
                max_workers=self.settings.max_workers,
                stop_on_solved=self.settings.stop_on_solved,
                agentic_roles=self.settings.agentic_roles,
                reuse_role_sessions=self.settings.reuse_role_sessions,
                model=self.settings.model,
                effort=self.settings.effort,
                crow_enabled=self.settings.crow_enabled,
                crow_max_nudges=self.settings.crow_max_nudges,
                crow_event_limit=self.settings.crow_event_limit,
                helm_directives=helm_directives,
                context_policy=ContextPolicy(
                    max_parent_summary_chars=self.settings.max_parent_context_chars,
                    max_worker_output_chars=self.settings.max_worker_context_chars,
                    keep_raw_outputs=self.settings.keep_raw_outputs,
                ),
            ),
            role_command=self.settings.role_command,
        )
        result = orchestrator.run(
            goal=goal,
            command=self.settings.command,
            cwd=self._effective_cwd(),
            seed_files=seed_files,
            metadata={
                "backend": self.settings.backend,
                "helm_directives": [asdict(directive) for directive in helm_directives],
                "cube_diagnosis": format_cube_diagnosis(cube_diagnosis) if cube_diagnosis else None,
                "source": "tui",
            },
            resume_run=resume_run,
        )
        payload = {"backend": self.settings.backend, **result.as_dict()}
        self.last_payload = payload
        best = result.best_candidate
        self.status_line = f"run finished: {result.run.run_id}"
        print(f"run finished: run={result.run.run_id} candidates={len(result.candidates)}")
        if best:
            print(f"best: {best.status} score={best.score:.3f} seed={best.seed.label}")
            print(best.summary)
        print(f"result: {Path(result.run.root_dir) / 'result.json'}")

    def _prepare_runtime(self) -> tuple[AgentRuntime, CubeDiagnosis | None] | None:
        if self.settings.backend == "cube":
            result = _prepare_cube_backend(self._cube_namespace(), Path.cwd())
            if isinstance(result, int):
                return None
            return CubeSandboxRuntime(CubeSandboxConfig.from_env()), result
        try:
            return _build_runtime(self.settings.backend), None
        except RuntimeError as exc:
            print(f"{self.settings.backend}: {exc}")
            return None

    def _prepare_local_agent_commands(self) -> bool:
        if self.settings.backend != "codex":
            return True
        resolved, error = resolve_host_command(self.settings.command)
        if error:
            self.status_line = "run blocked: command missing"
            print(error)
            return False
        if resolved != self.settings.command:
            self.settings.command = resolved
            print(f"command = {resolved}")
        if self.settings.role_command:
            resolved_role, role_error = resolve_host_command(self.settings.role_command)
            if role_error:
                self.status_line = "run blocked: role command missing"
                print(role_error)
                return False
            if resolved_role != self.settings.role_command:
                self.settings.role_command = resolved_role
                print(f"role_command = {resolved_role}")
        return True

    def _cube_namespace(self) -> Any:
        class CubeArgs:
            pass

        args = CubeArgs()
        args.cube_mode = self.settings.cube_mode
        args.cube_api_url = self.settings.cube_api_url
        args.cube_dev_vm_api_url = self.settings.cube_dev_vm_api_url
        args.cube_template_id = self.settings.cube_template_id
        args.cube_template_image = self.settings.cube_template_image
        args.cube_wait_seconds = self.settings.cube_wait_seconds
        args.cube_no_install = not self.settings.cube_install
        args.cube_no_create_template = not self.settings.cube_create_template
        return args

    def _effective_cwd(self) -> str:
        return self.settings.cwd or ("/workspace" if self.settings.backend == "cube" else str(Path.cwd()))

    def _prompt(self, prompt: TextPrompt | None = None) -> str:
        if prompt:
            return f"pawahara:{prompt.label}> "
        backend = self.settings.backend
        return f"pawahara:{backend}> "

    def _print_banner(self) -> None:
        print("Pawahara Harness TUI")
        print("Type an instruction to run the harness. Use /help for commands and /settings for current config.")

    def _print_help(self, topic: str | None = None) -> None:
        if topic and normalize_key(topic) == "settings":
            print("Settings:")
            for key in sorted(SETTING_SPECS):
                print(f"  /{key.replace('_', '-')} <value>")
            print("Lists: /seed add <sandbox_path=local_path>, /helm add [scope:]text, /helm-file add [scope:]path")
            return
        print(
            "\n".join(
                [
                    "Commands:",
                    "  /run <instruction>       run the orchestrated harness",
                    "  /set <key> <value>       set any scalar setting",
                    "  /unset <key>             clear nullable settings like model/cwd/resume-run",
                    "  /toggle <key>            flip a boolean setting",
                    "  /seed add|list|remove|clear ...",
                    "  /helm add|list|remove|clear [scope:]text",
                    "  /helm-file add|list|remove|clear [scope:]path",
                    "  /cube status|up|use|settings",
                    "  /settings                show current config",
                    "  /last                    print last JSON payload",
                    "  /help settings           list every configurable key",
                    "  /exit                    quit",
                    "",
                    "Plain input is treated the same as /run <instruction>.",
                ]
            )
        )

    def _print_settings(self, prefix: str = "") -> None:
        payload = asdict(self.settings)
        for key in sorted(payload):
            if prefix and not key.startswith(prefix):
                continue
            print(f"{key} = {format_value(payload[key])}")
        if not prefix:
            print(f"effective_cwd = {self._effective_cwd()}")

    def _print_last_payload(self) -> None:
        if self.last_payload is None:
            print("No run result yet.")
            return
        print(json.dumps(self.last_payload, ensure_ascii=False, indent=2))

    def _print_list(self, name: str, values: list[str]) -> None:
        if not values:
            print(f"{name}: <empty>")
            return
        for index, value in enumerate(values):
            print(f"{index}: {value}")


def normalize_key(value: str) -> str:
    return value.strip().lstrip("-").replace("-", "_")


def is_command_line(value: str) -> bool:
    return value.startswith("\\") or value.startswith("/")


def command_catalog() -> tuple[CommandSuggestion, ...]:
    suggestions = [CommandSuggestion(name, description, " ") for name, description in CORE_COMMANDS]
    core_names = {suggestion.name for suggestion in suggestions}
    for key, spec in sorted(SETTING_SPECS.items()):
        name = key.replace("_", "-")
        if name in core_names:
            continue
        suggestions.append(CommandSuggestion(name, describe_setting(spec), " "))
    return tuple(suggestions)


def describe_setting(spec: SettingSpec) -> str:
    if spec.choices:
        return "set " + spec.attr.replace("_", "-") + " (" + "|".join(spec.choices) + ")"
    if spec.kind == "bool":
        return "toggle " + spec.attr.replace("_", "-")
    return "set " + spec.attr.replace("_", "-")


def command_suggestions_for_input(text: str) -> list[CommandSuggestion]:
    parsed = parse_slash_input(text)
    if parsed is None:
        return []
    if parsed.arg_text is not None:
        return argument_suggestions_for_input(parsed)
    token = parsed.command_token
    catalog = command_catalog()
    if not token:
        return list(catalog[:12])
    return filter_suggestions(catalog, token, normalize_names=True)


def suggestion_group_for_input(text: str) -> str:
    parsed = parse_slash_input(text)
    if parsed is not None and parsed.arg_text is not None:
        return "choices"
    return "commands"


def argument_suggestions_for_input(parsed: SlashInput) -> list[CommandSuggestion]:
    command = normalize_key(parsed.command_token)
    arg_text = parsed.arg_text or ""
    if not command:
        return []
    if command == "set":
        return set_command_suggestions(arg_text)
    if command == "unset":
        return setting_key_suggestions(arg_text, nullable_only=True)
    if command == "toggle":
        return setting_key_suggestions(arg_text, bool_only=True)
    if command in {"seed", "helm", "helm_file"}:
        return list_action_suggestions(arg_text)
    if command == "cube":
        return cube_action_suggestions(arg_text)
    if command == "help":
        return filter_suggestions((CommandSuggestion("settings", "list configurable keys"),), arg_text)
    if command in SETTING_SPECS:
        return setting_value_suggestions(command, arg_text)
    return []


def set_command_suggestions(arg_text: str) -> list[CommandSuggestion]:
    context = argument_context(arg_text)
    if len(context.parts) == 0:
        return setting_key_suggestions(context.current, value_required=True)
    if len(context.parts) == 1 and not context.trailing_space:
        return setting_key_suggestions(context.current, value_required=True)
    if len(context.parts) == 1 and context.trailing_space:
        return setting_value_suggestions(context.parts[0], "")
    if len(context.parts) == 2 and not context.trailing_space:
        return setting_value_suggestions(context.parts[0], context.current)
    return []


def setting_key_suggestions(
    prefix: str,
    *,
    nullable_only: bool = False,
    bool_only: bool = False,
    value_required: bool = False,
) -> list[CommandSuggestion]:
    context = argument_context(prefix)
    if len(context.parts) > 1 or (len(context.parts) == 1 and context.trailing_space):
        return []
    current = context.current
    suggestions = []
    for key, spec in sorted(SETTING_SPECS.items()):
        if nullable_only and not spec.nullable:
            continue
        if bool_only and spec.kind != "bool":
            continue
        suffix = " " if value_required else ""
        suggestions.append(CommandSuggestion(key.replace("_", "-"), describe_setting(spec), suffix))
    return filter_suggestions(tuple(suggestions), current, normalize_names=True)


def setting_value_suggestions(key: str, prefix: str) -> list[CommandSuggestion]:
    normalized_key = normalize_key(key)
    spec = SETTING_SPECS.get(normalized_key)
    if not spec:
        return []
    context = argument_context(prefix)
    if len(context.parts) > 1 or (len(context.parts) == 1 and context.trailing_space):
        return []
    current = context.current
    if spec.choices:
        candidates = tuple(CommandSuggestion(choice, f"set {spec.attr.replace('_', '-')}") for choice in spec.choices)
        return filter_suggestions(candidates, current)
    if normalized_key == "effort":
        candidates = tuple(CommandSuggestion(choice, "reasoning effort") for choice in EFFORT_CHOICES)
        return filter_suggestions(candidates, current)
    if normalized_key == "model":
        candidates = tuple(CommandSuggestion(choice, "model suggestion") for choice in MODEL_SUGGESTIONS)
        return filter_suggestions(candidates, current)
    if spec.kind == "bool":
        candidates = tuple(CommandSuggestion(choice, f"set {spec.attr.replace('_', '-')}") for choice in BOOLEAN_CHOICES)
        return filter_suggestions(candidates, current)
    return []


def list_action_suggestions(arg_text: str) -> list[CommandSuggestion]:
    context = argument_context(arg_text)
    if len(context.parts) > 1 or (len(context.parts) == 1 and context.trailing_space):
        return []
    descriptions = {
        "add": "append a value",
        "list": "show values",
        "remove": "remove by index",
        "clear": "remove all values",
    }
    candidates = tuple(
        CommandSuggestion(action, descriptions[action], " " if action in {"add", "remove"} else "")
        for action in LIST_ACTIONS
    )
    return filter_suggestions(candidates, context.current)


def cube_action_suggestions(arg_text: str) -> list[CommandSuggestion]:
    context = argument_context(arg_text)
    if len(context.parts) > 1 or (len(context.parts) == 1 and context.trailing_space):
        return []
    descriptions = {
        "status": "check CubeSandbox API",
        "up": "start or discover CubeSandbox",
        "use": "set backend to cube",
        "settings": "show cube settings",
    }
    candidates = tuple(CommandSuggestion(action, descriptions[action]) for action in CUBE_ACTIONS)
    return filter_suggestions(candidates, context.current)


def filter_suggestions(
    suggestions: tuple[CommandSuggestion, ...],
    prefix: str,
    *,
    normalize_names: bool = False,
) -> list[CommandSuggestion]:
    needle = prefix.strip().lower()
    if normalize_names:
        needle = normalize_key(needle).replace("_", "-")
    if not needle:
        return list(suggestions)

    def comparison_name(suggestion: CommandSuggestion) -> str:
        value = suggestion.name.lower()
        if normalize_names:
            return normalize_key(value).replace("_", "-")
        return value

    exact = [suggestion for suggestion in suggestions if comparison_name(suggestion) == needle]
    prefixed = [
        suggestion
        for suggestion in suggestions
        if comparison_name(suggestion).startswith(needle) and comparison_name(suggestion) != needle
    ]
    return exact + prefixed


def argument_context(arg_text: str) -> ArgumentContext:
    trailing_space = bool(arg_text) and arg_text[-1].isspace()
    parts = tuple(arg_text.split())
    current = "" if trailing_space or not parts else parts[-1]
    return ArgumentContext(parts=parts, current=current, trailing_space=trailing_space)


def replace_current_argument(parsed: SlashInput, suggestion: CommandSuggestion) -> str:
    arg_text = parsed.arg_text or ""
    command = f"{parsed.prefix}{parsed.command_token}"
    replacement = f"{suggestion.name}{suggestion.completion_suffix}"
    if not arg_text:
        return f"{command} {replacement}"
    if arg_text[-1].isspace():
        return f"{command} {arg_text}{replacement}"
    head, separator, _current = arg_text.rpartition(" ")
    if separator:
        return f"{command} {head} {replacement}"
    return f"{command} {replacement}"


def visible_suggestions(
    suggestions: list[CommandSuggestion],
    selected: int,
    max_visible: int = MAX_VISIBLE_SUGGESTIONS,
) -> tuple[int, list[CommandSuggestion]]:
    if not suggestions:
        return 0, []
    selected = max(0, min(selected, len(suggestions) - 1))
    max_visible = max(1, max_visible)
    if len(suggestions) <= max_visible:
        return 0, suggestions
    half_window = max_visible // 2
    start = selected - half_window
    start = max(0, min(start, len(suggestions) - max_visible))
    return start, suggestions[start : start + max_visible]


def parse_slash_input(text: str) -> SlashInput | None:
    if not text or text[0] not in {"/", "\\"}:
        return None
    rest = text[1:]
    if rest.startswith(" "):
        return None
    if not rest:
        return SlashInput(text[0], "", None)
    command_token, separator, arg_text = rest.partition(" ")
    if not separator:
        return SlashInput(text[0], command_token, None)
    return SlashInput(text[0], command_token, arg_text)


def slash_edit_token(text: str) -> tuple[str, str] | None:
    parsed = parse_slash_input(text)
    if parsed is None or parsed.arg_text is not None:
        return None
    return parsed.prefix, parsed.command_token


def complete_command_input(text: str, selected: int) -> str | None:
    parsed = parse_slash_input(text)
    if parsed is None:
        return None
    suggestions = command_suggestions_for_input(text)
    if not suggestions:
        return None
    selected = max(0, min(selected, len(suggestions) - 1))
    suggestion = suggestions[selected]
    if parsed.arg_text is None:
        return f"{parsed.prefix}{suggestion.name}{suggestion.completion_suffix}"
    return replace_current_argument(parsed, suggestion)


def read_escape_action() -> str:
    sequence = ""
    for _ in range(8):
        char = read_optional_stdin_char(timeout_tenths=2)
        if not char:
            break
        sequence += char
        if sequence and sequence[-1].isalpha():
            break
    return parse_escape_action(sequence)


def read_optional_stdin_char(timeout_tenths: int = 1) -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[6][termios.VMIN] = 0
    new_settings[6][termios.VTIME] = max(0, timeout_tenths)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def parse_escape_action(sequence: str) -> str:
    if not sequence:
        return "escape"
    if sequence in {"[A", "OA"}:
        return "up"
    if sequence in {"[B", "OB"}:
        return "down"
    if sequence.endswith("A") and sequence.startswith("["):
        return "up"
    if sequence.endswith("B") and sequence.startswith("["):
        return "down"
    return "escape"


def truncate_for_status(value: str, max_chars: int = 72) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def highlight_prompt(value: str) -> str:
    prompt = value.rstrip(" ")
    suffix = value[len(prompt) :]
    return f"{ANSI_CURRENT_LINE}{prompt}{ANSI_RESET}{suffix}"


def format_prompt_line(prompt: str, text: str) -> str:
    return f"{highlight_prompt(prompt)}{text}"


def resolve_host_command(
    command: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    discover_codex: Callable[[], Path | None] | None = None,
) -> tuple[str, str | None]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return command, f"Could not parse agent command: {exc}"
    if not parts:
        return command, "Agent command is empty. Set it with /command <shell command>."

    executable = parts[0]
    if "/" in executable:
        if Path(executable).exists():
            return normalize_codex_exec_command(parts), None
        return command, f"Agent command executable not found: {executable}"

    if which(executable):
        return normalize_codex_exec_command(parts), None

    if executable == "codex":
        codex_path = (discover_codex or discover_codex_cli)()
        if codex_path:
            return normalize_codex_exec_command((str(codex_path), *parts[1:])), None

    return command, (
        f"Agent command executable `{executable}` was not found in PATH.\n"
        f"Current command: {command}\n"
        "Use /backend codex-sdk, or set an absolute command with /command <path-to-codex> exec ..."
    )


def normalize_codex_exec_command(parts: tuple[str, ...] | list[str]) -> str:
    if not parts:
        return ""
    normalized = list(parts)
    executable = Path(normalized[0]).name
    if executable != "codex" or "exec" not in normalized:
        return shell_join(tuple(normalized))
    if "--ask-for-approval" in normalized or "-a" in normalized:
        return shell_join(tuple(normalized))
    if "--approval-policy" not in normalized:
        return shell_join(tuple(normalized))

    policy_index = normalized.index("--approval-policy")
    if policy_index + 1 >= len(normalized):
        return shell_join(tuple(normalized))
    policy = normalized[policy_index + 1]
    del normalized[policy_index : policy_index + 2]
    exec_index = normalized.index("exec")
    normalized[exec_index:exec_index] = ["--ask-for-approval", policy]
    return shell_join(tuple(normalized))


def discover_codex_cli() -> Path | None:
    found = shutil.which("codex")
    if found:
        return Path(found)
    candidates: list[Path] = []
    vscode_extensions = Path.home() / ".vscode" / "extensions"
    if vscode_extensions.exists():
        candidates.extend(vscode_extensions.glob("openai.chatgpt-*/bin/*/codex"))
    cursor_extensions = Path.home() / ".cursor" / "extensions"
    if cursor_extensions.exists():
        candidates.extend(cursor_extensions.glob("openai.chatgpt-*/bin/*/codex"))
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def shell_join(parts: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def setting_requires_value(spec: SettingSpec) -> bool:
    if spec.kind == "bool":
        return False
    if spec.attr == "backend" and spec.choices == ("cube",):
        return False
    return True


def parse_setting_value(spec: SettingSpec, raw_value: str | None) -> Any:
    if spec.kind == "bool":
        parsed = True if raw_value is None or raw_value == "" else parse_bool(raw_value)
        return not parsed if spec.inverted else parsed
    if raw_value is None or raw_value == "":
        if spec.attr == "backend" and spec.choices == ("cube",):
            return "cube"
        if spec.nullable:
            return None
        raise ValueError(f"{spec.attr} requires a value.")
    if spec.kind == "int":
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{spec.attr} must be an integer: {raw_value}") from exc
        if value < 0:
            raise ValueError(f"{spec.attr} must be non-negative.")
        return value
    if spec.choices:
        value = raw_value.strip()
        if spec.attr == "backend" and spec.choices == ("cube",):
            return "cube"
        if value not in spec.choices:
            raise ValueError(f"{spec.attr} must be one of: {', '.join(spec.choices)}")
        return value
    return raw_value


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def format_value(value: Any) -> str:
    if value is None:
        return "<auto>"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_cube_diagnosis(diagnosis: CubeDiagnosis) -> dict[str, Any]:
    return diagnosis.__dict__ | {
        "environment": diagnosis.environment.as_env() if diagnosis.environment else None,
    }
