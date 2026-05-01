from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_TEMPLATE_IMAGE = "ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest"
ENV_FILE = Path(".pawahara/cube.env")


@dataclass(frozen=True)
class CubeEnvironment:
    api_url: str
    api_key: str
    template_id: str
    ssl_cert_file: str | None = None

    def as_env(self) -> dict[str, str]:
        env = {
            "E2B_API_URL": self.api_url,
            "E2B_API_KEY": self.api_key,
            "CUBE_TEMPLATE_ID": self.template_id,
        }
        if self.ssl_cert_file:
            env["SSL_CERT_FILE"] = self.ssl_cert_file
        return env


@dataclass(frozen=True)
class CubeBootstrapOptions:
    repo_root: Path
    api_url: str = "http://127.0.0.1:3000"
    dev_vm_api_url: str = "http://127.0.0.1:13000"
    api_key: str = "dummy"
    template_id: str | None = None
    template_image: str = DEFAULT_TEMPLATE_IMAGE
    mode: str = "auto"
    wait_seconds: int = 900
    install: bool = True
    create_template: bool = True


@dataclass(frozen=True)
class CubeDiagnosis:
    ok: bool
    mode: str
    reason: str
    environment: CubeEnvironment | None = None


class CubeBootstrapper:
    def __init__(self, options: CubeBootstrapOptions):
        self.options = options
        self.cubesandbox_dir = options.repo_root / "CubeSandbox"
        self.dev_env_dir = self.cubesandbox_dir / "dev-env"

    def status(self) -> CubeEnvironment | None:
        file_env = read_cube_env(self.options.repo_root / ENV_FILE)
        template_id = self.options.template_id or file_env.get("CUBE_TEMPLATE_ID")

        for api_url in candidate_api_urls(self.options, file_env):
            if cube_api_healthy(api_url):
                if template_id:
                    return CubeEnvironment(
                        api_url=api_url,
                        api_key=file_env.get("E2B_API_KEY", self.options.api_key),
                        template_id=template_id,
                        ssl_cert_file=file_env.get("SSL_CERT_FILE"),
                    )
                return CubeEnvironment(
                    api_url=api_url,
                    api_key=file_env.get("E2B_API_KEY", self.options.api_key),
                    template_id="",
                    ssl_cert_file=file_env.get("SSL_CERT_FILE"),
                )
        return None

    def up(self) -> CubeEnvironment:
        existing = self.status()
        if existing and existing.template_id:
            write_cube_env(self.options.repo_root / ENV_FILE, existing)
            return existing

        mode = self._resolve_mode()
        if mode == "direct":
            env = self._up_direct()
        elif mode == "dev-vm":
            env = self._up_dev_vm()
        else:
            raise RuntimeError(f"unsupported CubeSandbox bootstrap mode: {mode}")

        write_cube_env(self.options.repo_root / ENV_FILE, env)
        return env

    def diagnose(self) -> CubeDiagnosis:
        existing = self.status()
        if existing and existing.template_id:
            return CubeDiagnosis(
                ok=True,
                mode="existing",
                reason=f"Cube API is healthy at {existing.api_url} and template id is configured.",
                environment=existing,
            )
        if existing and not self.options.create_template and not self.options.template_id:
            return CubeDiagnosis(
                ok=False,
                mode="existing",
                reason=f"Cube API is healthy at {existing.api_url}, but no template id is configured.",
                environment=existing,
            )

        mode = self._resolve_mode()
        if mode == "direct":
            if not is_linux_with_kvm():
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason="direct CubeSandbox requires Linux with /dev/kvm.",
                    environment=existing,
                )
            if not self.options.install and not existing:
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason="Cube API is not healthy and --no-install was set.",
                    environment=existing,
                )
            if not self.options.template_id and not self.options.create_template:
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason="no template id is configured and template creation is disabled.",
                    environment=existing,
                )
            return CubeDiagnosis(
                ok=True,
                mode=mode,
                reason="Linux/KVM direct CubeSandbox startup looks viable.",
                environment=existing,
            )

        if mode == "dev-vm":
            if platform.system() != "Linux":
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason=(
                        "CubeSandbox dev VM requires a Linux host with KVM. "
                        "Use an existing Linux CubeSandbox endpoint from macOS."
                    ),
                    environment=existing,
                )
            if not is_linux_with_kvm():
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason="CubeSandbox dev VM requires /dev/kvm and nested virtualization.",
                    environment=existing,
                )
            if not self.dev_env_dir.exists():
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason=f"CubeSandbox dev-env directory is missing: {self.dev_env_dir}",
                    environment=existing,
                )
            if not self.options.template_id and not self.options.create_template:
                return CubeDiagnosis(
                    ok=False,
                    mode=mode,
                    reason="no template id is configured and template creation is disabled.",
                    environment=existing,
                )
            return CubeDiagnosis(
                ok=True,
                mode=mode,
                reason="Linux/KVM dev VM CubeSandbox startup looks viable.",
                environment=existing,
            )

        return CubeDiagnosis(
            ok=False,
            mode=mode,
            reason=f"unsupported CubeSandbox bootstrap mode: {mode}",
            environment=existing,
        )

    def _resolve_mode(self) -> str:
        if self.options.mode != "auto":
            return self.options.mode
        if is_linux_with_kvm():
            return "direct"
        return "dev-vm"

    def _up_direct(self) -> CubeEnvironment:
        if not is_linux_with_kvm():
            raise RuntimeError("direct CubeSandbox startup requires Linux with /dev/kvm.")
        if self.options.install and not cube_api_healthy(self.options.api_url):
            require_command("curl")
            run_checked(
                [
                    "bash",
                    "-lc",
                    "curl -sL https://github.com/tencentcloud/CubeSandbox/raw/master/deploy/one-click/online-install.sh | bash",
                ],
                cwd=self.options.repo_root,
            )
            wait_for_health(self.options.api_url, self.options.wait_seconds)

        template_id = self.options.template_id
        if not template_id and self.options.create_template:
            template_id = self._create_template_local()
        if not template_id:
            raise RuntimeError("CubeSandbox is running, but no template id is configured.")
        return CubeEnvironment(api_url=self.options.api_url, api_key=self.options.api_key, template_id=template_id)

    def _up_dev_vm(self) -> CubeEnvironment:
        if platform.system() != "Linux":
            raise RuntimeError(
                "CubeSandbox dev VM automation requires a Linux host with KVM. "
                "On macOS, run this harness against a remote/bare-metal Linux CubeSandbox "
                "by setting E2B_API_URL and CUBE_TEMPLATE_ID, or run the dev-env on a Linux machine."
            )
        if not is_linux_with_kvm():
            raise RuntimeError("CubeSandbox dev VM requires Linux with /dev/kvm and nested virtualization.")

        if not cube_api_healthy(self.options.dev_vm_api_url):
            self._prepare_dev_vm_image_if_needed()
            self._start_dev_vm_background()
            self._wait_for_ssh()
            if self.options.install:
                self._install_inside_dev_vm()
            wait_for_health(self.options.dev_vm_api_url, self.options.wait_seconds)

        template_id = self.options.template_id
        if not template_id and self.options.create_template:
            template_id = self._create_template_in_dev_vm()
        if not template_id:
            raise RuntimeError("CubeSandbox dev VM is running, but no template id is configured.")
        return CubeEnvironment(
            api_url=self.options.dev_vm_api_url,
            api_key=self.options.api_key,
            template_id=template_id,
            ssl_cert_file=None,
        )

    def _prepare_dev_vm_image_if_needed(self) -> None:
        workdir = self.dev_env_dir / ".workdir"
        if any(workdir.glob("*.qcow2")):
            return
        run_checked(["./prepare_image.sh"], cwd=self.dev_env_dir)

    def _start_dev_vm_background(self) -> None:
        pidfile = self.dev_env_dir / ".workdir" / "qemu.pid"
        if pidfile.exists() and process_alive(pidfile.read_text().strip()):
            return
        env = os.environ.copy()
        env["VM_BACKGROUND"] = "1"
        run_checked(["./run_vm.sh"], cwd=self.dev_env_dir, env=env)

    def _wait_for_ssh(self) -> None:
        deadline = time.time() + self.options.wait_seconds
        while time.time() < deadline:
            try:
                self._ssh("true", check=True)
                return
            except subprocess.CalledProcessError:
                time.sleep(3)
        raise RuntimeError("timed out waiting for CubeSandbox dev VM SSH.")

    def _install_inside_dev_vm(self) -> None:
        if self._ssh("curl -sf http://127.0.0.1:3000/health >/dev/null", check=False).returncode == 0:
            return
        install_cmd = (
            "curl -sL https://github.com/tencentcloud/CubeSandbox/raw/master/deploy/one-click/online-install.sh | bash"
        )
        self._ssh_root(install_cmd)
        self._ssh_root("systemctl enable --now cube-sandbox-oneclick.service || true")

    def _create_template_local(self) -> str:
        require_command("cubemastercli")
        create_output = run_text(
            [
                "cubemastercli",
                "tpl",
                "create-from-image",
                "--image",
                self.options.template_image,
                "--writable-layer-size",
                "1G",
                "--expose-port",
                "49999",
                "--expose-port",
                "49983",
                "--probe",
                "49999",
            ],
            cwd=self.options.repo_root,
        )
        job_id = extract_first(create_output, [r"job[_ -]?id[:=\s]+([A-Za-z0-9._-]+)"])
        if job_id:
            watch_output = run_text(["cubemastercli", "tpl", "watch", "--job-id", job_id], cwd=self.options.repo_root)
            return extract_template_id(watch_output)
        return extract_template_id(create_output)

    def _create_template_in_dev_vm(self) -> str:
        create_cmd = (
            "cubemastercli tpl create-from-image "
            f"--image {shell_quote(self.options.template_image)} "
            "--writable-layer-size 1G --expose-port 49999 --expose-port 49983 --probe 49999"
        )
        create_output = self._ssh_root_capture(create_cmd)
        job_id = extract_first(create_output, [r"job[_ -]?id[:=\s]+([A-Za-z0-9._-]+)"])
        if job_id:
            watch_output = self._ssh_root_capture(f"cubemastercli tpl watch --job-id {shell_quote(job_id)}")
            return extract_template_id(watch_output)
        return extract_template_id(create_output)

    def _ssh(self, command: str, *, check: bool) -> subprocess.CompletedProcess[str]:
        return ssh_with_password(
            command,
            cwd=self.dev_env_dir,
            password=os.environ.get("VM_PASSWORD", "opencloudos"),
            check=check,
        )

    def _ssh_root(self, command: str) -> None:
        self._ssh(root_command(command), check=True)

    def _ssh_root_capture(self, command: str) -> str:
        return self._ssh(root_command(command), check=True).stdout


def candidate_api_urls(options: CubeBootstrapOptions, file_env: Mapping[str, str]) -> list[str]:
    urls = [
        file_env.get("E2B_API_URL", ""),
        os.environ.get("E2B_API_URL", ""),
        options.api_url,
        options.dev_vm_api_url,
    ]
    result: list[str] = []
    for url in urls:
        if url and url not in result:
            result.append(url)
    return result


def cube_api_healthy(api_url: str, timeout: float = 2.0) -> bool:
    for path in ("/health", "/v1/health"):
        try:
            with urllib.request.urlopen(api_url.rstrip("/") + path, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return False


def wait_for_health(api_url: str, seconds: int) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if cube_api_healthy(api_url):
            return
        time.sleep(3)
    raise RuntimeError(f"timed out waiting for CubeAPI at {api_url}")


def is_linux_with_kvm() -> bool:
    return platform.system() == "Linux" and Path("/dev/kvm").exists()


def process_alive(pid: str) -> bool:
    if not pid.isdigit():
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def require_command(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"required command not found: {name}")


def run_checked(
    args: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    subprocess.run(args, cwd=cwd, env=dict(env) if env else None, check=True)


def run_text(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout + completed.stderr


def ssh_with_password(
    command: str,
    *,
    cwd: Path,
    password: str,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    require_command("ssh")
    require_command("setsid")
    askpass = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        askpass.write("#!/usr/bin/env bash\n")
        askpass.write(f"printf '%s\\n' {shell_quote(password)}\n")
        askpass.close()
        os.chmod(askpass.name, 0o700)
        env = os.environ.copy()
        env["DISPLAY"] = env.get("DISPLAY", "cubesandbox-dev-env")
        env["SSH_ASKPASS"] = askpass.name
        env["SSH_ASKPASS_REQUIRE"] = "force"
        return subprocess.run(
            [
                "setsid",
                "-w",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-p",
                os.environ.get("SSH_PORT", "10022"),
                f"{os.environ.get('VM_USER', 'opencloudos')}@{os.environ.get('SSH_HOST', '127.0.0.1')}",
                command,
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=check,
        )
    finally:
        Path(askpass.name).unlink(missing_ok=True)


def root_command(command: str) -> str:
    password = os.environ.get("VM_PASSWORD", "opencloudos")
    return f"printf '%s\\n' {shell_quote(password)} | sudo -S bash -lc {shell_quote(command)}"


def extract_template_id(output: str) -> str:
    template_id = extract_first(
        output,
        [
            r"template[_ -]?id[:=\s]+([A-Za-z0-9._-]+)",
            r"\b(tpl_[A-Za-z0-9._-]+)\b",
        ],
    )
    if not template_id:
        raise RuntimeError(f"could not find template id in CubeSandbox output:\n{output}")
    return template_id


def extract_first(output: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def write_cube_env(path: Path, env: CubeEnvironment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={shell_quote(value)}" for key, value in env.as_env().items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_cube_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed = shlex.split(value, comments=False, posix=True)
        result[key] = parsed[0] if parsed else ""
    return result


def shell_quote(value: str) -> str:
    return shlex.quote(value)
