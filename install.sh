#!/usr/bin/env bash
set -euo pipefail

repo="${PAWAHARA_REPO:-mizuamedesu/PawaharaHarness}"
version="${PAWAHARA_VERSION:-latest}"
install_dir="${PAWAHARA_INSTALL_DIR:-$HOME/.local/bin}"
download_url="${PAWAHARA_DOWNLOAD_URL:-}"
binary_name="pawahara-harness"
install_skills="${PAWAHARA_INSTALL_SKILLS:-1}"

usage() {
  cat <<'EOF'
Install Pawahara Harness on Linux.

Usage:
  ./install.sh [--version v0.1.0] [--install-dir ~/.local/bin]

Environment:
  PAWAHARA_REPO          GitHub repo in owner/name form. Default: mizuamedesu/PawaharaHarness
  PAWAHARA_VERSION       Release tag to install. Default: latest
  PAWAHARA_INSTALL_DIR   Directory for the binary. Default: ~/.local/bin
  PAWAHARA_DOWNLOAD_URL  Override archive URL.
  PAWAHARA_INSTALL_SKILLS Install Codex and Claude Code skills. Default: 1
  CODEX_HOME             Codex home directory. Default: ~/.codex
  CLAUDE_HOME            Claude Code home directory. Default: ~/.claude

Examples:
  curl -fsSL https://raw.githubusercontent.com/mizuamedesu/PawaharaHarness/main/install.sh | bash
  PAWAHARA_VERSION=v0.1.0 ./install.sh
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

expand_path() {
  case "$1" in
    "~")
      printf '%s\n' "$HOME"
      ;;
    "~/"*)
      printf '%s/%s\n' "$HOME" "${1#~/}"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 --output "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    die "curl or wget is required"
  fi
}

path_entry_for_shell() {
  case "$install_dir" in
    "$HOME")
      printf '$HOME'
      ;;
    "$HOME"/*)
      printf '$HOME/%s' "${install_dir#"$HOME"/}"
      ;;
    *)
      printf '%s' "$install_dir"
      ;;
  esac
}

append_path_to_rc() {
  local rc_file="$1"
  local path_entry="$2"
  local marker_begin="# >>> pawahara-harness installer >>>"
  local marker_end="# <<< pawahara-harness installer <<<"

  mkdir -p "$(dirname "$rc_file")"
  touch "$rc_file"
  if grep -Fq "$marker_begin" "$rc_file"; then
    return
  fi

  {
    printf '\n%s\n' "$marker_begin"
    printf 'case ":$PATH:" in\n'
    printf '  *":%s:"*) ;;\n' "$path_entry"
    printf '  *) export PATH="%s:$PATH" ;;\n' "$path_entry"
    printf 'esac\n'
    printf '%s\n' "$marker_end"
  } >>"$rc_file"
}

ensure_path() {
  case ":$PATH:" in
    *":$install_dir:"*)
      return
      ;;
  esac

  local path_entry
  path_entry="$(path_entry_for_shell)"
  append_path_to_rc "$HOME/.profile" "$path_entry"

  if [ -f "$HOME/.bashrc" ] || [ "${SHELL##*/}" = "bash" ]; then
    append_path_to_rc "$HOME/.bashrc" "$path_entry"
  fi
  if [ -f "$HOME/.zshrc" ] || [ "${SHELL##*/}" = "zsh" ]; then
    append_path_to_rc "$HOME/.zshrc" "$path_entry"
  fi

  printf 'Added %s to PATH in your shell profile.\n' "$install_dir"
  printf 'Restart your shell or run: export PATH="%s:$PATH"\n' "$install_dir"
}

write_codex_skill() {
  local skill_dir="$1"
  mkdir -p "$skill_dir"
  cat >"$skill_dir/SKILL.md" <<'EOF'
---
name: pawahara-harness
description: Use Pawahara Harness for difficult coding, CTF, competitive programming, debugging, optimization, or long-running tasks that benefit from diverse beam-search workers, scoring, resume, Karasu continuation, and the monitor UI.
metadata:
  short-description: Run Pawahara Harness search
---

# Pawahara Harness

Use this skill when the user asks to use Pawahara Harness, wants more diverse reasoning, asks for CTF or competitive-programming solving, needs multi-worker exploration, or the current agent is likely to stop too early.

## Run Search

Prefer `search` for real work:

```sh
goal="$(cat <<'GOAL'
<user task>
GOAL
)"
pawahara-harness search \
  --goal "$goal" \
  --cwd "$(pwd)" \
  --beam-width 4 \
  --branch-factor 4 \
  --max-depth 2 \
  --max-workers 4
```

For a quick smoke test or a single direct agent, use:

```sh
pawahara-harness run --goal "$goal" --cwd "$(pwd)"
```

## Resume

Continue an existing run instead of starting over:

```sh
pawahara-harness search \
  --resume-run <run_id_or_path> \
  --resume-message "$goal" \
  --cwd "$(pwd)"
```

## Steering

Use Helm for forced context:

```sh
pawahara-harness search \
  --goal "$goal" \
  --cwd "$(pwd)" \
  --helm "worker:<extra worker instruction>" \
  --helm "manager:<extra manager instruction>"
```

## Operating Notes

- Check `command -v pawahara-harness` before relying on the skill.
- Keep the user's original objective intact in `--goal`; do not summarize away constraints.
- Use `--runs-dir` when you need an isolated test run.
- Inspect `.pawahara/runs/<run_id>/result.json` and worker responses before reporting.
- Karasu is enabled by default; leave it on when the user wants persistence.
- Do not use the harness for tiny one-step edits unless the user explicitly asks.
EOF
}

write_claude_skill() {
  local skill_dir="$1"
  mkdir -p "$skill_dir"
  cat >"$skill_dir/SKILL.md" <<'EOF'
---
name: pawahara-harness
description: Use Pawahara Harness for difficult coding, CTF, competitive programming, debugging, optimization, or long-running tasks that benefit from diverse beam-search workers, scoring, resume, Karasu continuation, and the monitor UI.
argument-hint: "<task>"
---

# Pawahara Harness

Use this skill when the user asks to use Pawahara Harness, wants more diverse reasoning, asks for CTF or competitive-programming solving, needs multi-worker exploration, or the current agent is likely to stop too early.

If invoked with arguments, treat `$ARGUMENTS` as the task. Otherwise use the current user request.

## Run Search

Prefer `search` for real work:

```sh
goal="$(cat <<'GOAL'
$ARGUMENTS
GOAL
)"
pawahara-harness search \
  --goal "$goal" \
  --cwd "$(pwd)" \
  --beam-width 4 \
  --branch-factor 4 \
  --max-depth 2 \
  --max-workers 4
```

For a quick smoke test or a single direct agent, use:

```sh
pawahara-harness run --goal "$goal" --cwd "$(pwd)"
```

## Resume

Continue an existing run instead of starting over:

```sh
pawahara-harness search \
  --resume-run <run_id_or_path> \
  --resume-message "$goal" \
  --cwd "$(pwd)"
```

## Steering

Use Helm for forced context:

```sh
pawahara-harness search \
  --goal "$goal" \
  --cwd "$(pwd)" \
  --helm "worker:<extra worker instruction>" \
  --helm "manager:<extra manager instruction>"
```

## Operating Notes

- Check `command -v pawahara-harness` before relying on the skill.
- Keep the user's original objective intact in `--goal`; do not summarize away constraints.
- Use `--runs-dir` when you need an isolated test run.
- Inspect `.pawahara/runs/<run_id>/result.json` and worker responses before reporting.
- Karasu is enabled by default; leave it on when the user wants persistence.
- Do not use the harness for tiny one-step edits unless the user explicitly asks.
EOF
}

install_agent_skills() {
  case "$install_skills" in
    0|false|False|FALSE|no|No|NO)
      printf 'Skipped Codex and Claude Code skill registration.\n'
      return
      ;;
  esac

  local codex_home="${CODEX_HOME:-$HOME/.codex}"
  local claude_home="${CLAUDE_HOME:-$HOME/.claude}"
  local codex_skill_dir="$codex_home/skills/pawahara-harness"
  local claude_skill_dir="$claude_home/skills/pawahara-harness"

  write_codex_skill "$codex_skill_dir"
  write_claude_skill "$claude_skill_dir"

  printf 'Registered Codex skill: %s\n' "$codex_skill_dir/SKILL.md"
  printf 'Registered Claude Code skill: %s\n' "$claude_skill_dir/SKILL.md"
  printf 'Restart running Codex or Claude Code sessions if they do not pick up the new skill immediately.\n'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --version)
      [ "$#" -ge 2 ] || die "--version requires a value"
      version="$2"
      shift 2
      ;;
    --install-dir)
      [ "$#" -ge 2 ] || die "--install-dir requires a value"
      install_dir="$2"
      shift 2
      ;;
    --repo)
      [ "$#" -ge 2 ] || die "--repo requires a value"
      repo="$2"
      shift 2
      ;;
    --download-url)
      [ "$#" -ge 2 ] || die "--download-url requires a value"
      download_url="$2"
      shift 2
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[ "$(uname -s)" = "Linux" ] || die "this installer currently supports Linux only"

case "$(uname -m)" in
  x86_64|amd64)
    archive_name="pawahara-harness-linux-x86_64.tar.gz"
    ;;
  *)
    die "no prebuilt Linux binary is published for $(uname -m)"
    ;;
esac

install_dir="$(expand_path "$install_dir")"
mkdir -p "$install_dir"

if [ -z "$download_url" ]; then
  if [ "$version" = "latest" ]; then
    download_url="https://github.com/$repo/releases/latest/download/$archive_name"
  else
    download_url="https://github.com/$repo/releases/download/$version/$archive_name"
  fi
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive_path="$tmpdir/$archive_name"
printf 'Downloading %s\n' "$download_url"
download_file "$download_url" "$archive_path"

tar -xzf "$archive_path" -C "$tmpdir"
[ -f "$tmpdir/$binary_name" ] || die "archive did not contain $binary_name"

install -m 755 "$tmpdir/$binary_name" "$install_dir/$binary_name"
"$install_dir/$binary_name" --help >/dev/null

ensure_path
install_agent_skills
printf 'Installed %s to %s\n' "$binary_name" "$install_dir/$binary_name"
