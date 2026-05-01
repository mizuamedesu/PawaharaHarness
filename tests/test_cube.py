from __future__ import annotations

from pathlib import Path

import pawahara_harness.cube as cube
from pawahara_harness.cube import (
    CubeBootstrapOptions,
    CubeBootstrapper,
    CubeEnvironment,
    extract_template_id,
    read_cube_env,
    write_cube_env,
)


def test_cube_env_round_trips_shell_quoted_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".pawahara" / "cube.env"
    write_cube_env(
        env_path,
        CubeEnvironment(
            api_url="http://127.0.0.1:3000",
            api_key="dummy key",
            template_id="tpl_fake",
            ssl_cert_file="/tmp/cert path.pem",
        ),
    )

    assert read_cube_env(env_path) == {
        "E2B_API_URL": "http://127.0.0.1:3000",
        "E2B_API_KEY": "dummy key",
        "CUBE_TEMPLATE_ID": "tpl_fake",
        "SSL_CERT_FILE": "/tmp/cert path.pem",
    }


def test_extract_template_id_accepts_cli_output() -> None:
    assert extract_template_id("template id: tpl_abc123") == "tpl_abc123"
    assert extract_template_id("created tpl_xyz-789 successfully") == "tpl_xyz-789"


def test_status_discovers_saved_cube_endpoint(tmp_path: Path) -> None:
    write_cube_env(
        tmp_path / ".pawahara" / "cube.env",
        CubeEnvironment(
            api_url="http://saved-cube:3000",
            api_key="dummy",
            template_id="tpl_saved",
        ),
    )

    original = cube.cube_api_healthy
    try:
        cube.cube_api_healthy = lambda api_url, timeout=2.0: api_url == "http://saved-cube:3000"
        env = CubeBootstrapper(CubeBootstrapOptions(repo_root=tmp_path)).status()
    finally:
        cube.cube_api_healthy = original

    assert env is not None
    assert env.api_url == "http://saved-cube:3000"
    assert env.template_id == "tpl_saved"


def test_diagnose_accepts_existing_cube_endpoint(tmp_path: Path) -> None:
    write_cube_env(
        tmp_path / ".pawahara" / "cube.env",
        CubeEnvironment(
            api_url="http://saved-cube:3000",
            api_key="dummy",
            template_id="tpl_saved",
        ),
    )

    original = cube.cube_api_healthy
    try:
        cube.cube_api_healthy = lambda api_url, timeout=2.0: api_url == "http://saved-cube:3000"
        diagnosis = CubeBootstrapper(CubeBootstrapOptions(repo_root=tmp_path)).diagnose()
    finally:
        cube.cube_api_healthy = original

    assert diagnosis.ok
    assert diagnosis.mode == "existing"
    assert diagnosis.environment is not None
