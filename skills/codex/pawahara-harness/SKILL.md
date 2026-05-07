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
goal="$(cat <<'EOF'
<user task>
EOF
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
