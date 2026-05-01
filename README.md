# Pawahara Harness

Python harness for launching a main agent and subagents. The default runtime is
the local Codex CLI, so Codex keeps control of its own sandbox. CubeSandbox is
an explicit opt-in backend.

This project intentionally does not read or copy Codex credentials from the
host. If an agent command needs Codex ChatGPT login, make that available in the
CubeSandbox template or pass explicit environment/configuration yourself.

## Run An Agent

By default this runs `codex exec` locally and relies on Codex's own sandbox:

```bash
pawahara-harness run --goal "inspect this repository"
```

Override the command when needed:

```bash
pawahara-harness run \
  --goal "inspect this repository" \
  --command "codex exec --skip-git-repo-check --sandbox read-only --approval-policy never"
```

## CubeSandbox Backend

CubeSandbox exposes an E2B-compatible API. The harness can discover/start it
and write the needed SDK variables to `.pawahara/cube.env`:

```bash
pawahara-harness cube up
```

`cube up` first checks existing CubeAPI endpoints. On Linux with `/dev/kvm`, it
can run the one-click installer directly. On Linux dev machines it can also use
the CubeSandbox `dev-env` VM path. macOS cannot host CubeSandbox MicroVMs
directly; point the harness at a Linux CubeSandbox instance instead:

```bash
export E2B_API_URL="http://127.0.0.1:3000"
export E2B_API_KEY="dummy"
export CUBE_TEMPLATE_ID="<template-id>"
```

Manual status check:

```bash
pawahara-harness cube status
```

To switch a run to CubeSandbox, opt in explicitly. The harness diagnoses the
Cube backend first, then starts or discovers CubeSandbox only when the host
looks viable:

```bash
pawahara-harness run \
  --backend cube \
  --goal "inspect this repository" \
  --command "codex exec --skip-git-repo-check --sandbox danger-full-access --approval-policy never"
```

Programmatic use:

```python
from pawahara_harness import AgentLaunchSpec, CubeSandboxConfig, CubeSandboxRuntime

runtime = CubeSandboxRuntime(CubeSandboxConfig.from_env())

main = runtime.run_agent(
    AgentLaunchSpec(
        name="main",
        role="main",
        command="python3 - <<'PY'\nprint(input())\nPY",
        prompt="hello from harness",
    )
)
print(main.stdout)
```
