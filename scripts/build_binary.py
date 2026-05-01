from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a standalone Pawahara Harness binary with PyInstaller.")
    parser.add_argument("--name", default="pawahara-harness", help="Executable base name.")
    parser.add_argument("--dist-dir", default="dist", help="PyInstaller dist directory.")
    parser.add_argument("--work-dir", default="build/pyinstaller", help="PyInstaller work directory.")
    parser.add_argument("--spec-dir", default="build/pyinstaller-spec", help="PyInstaller spec directory.")
    parser.add_argument("--clean", action="store_true", help="Remove existing build output first.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = repo_root / args.dist_dir
    work_dir = repo_root / args.work_dir
    spec_dir = repo_root / args.spec_dir

    if args.clean:
        shutil.rmtree(dist_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(spec_dir, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--name",
        args.name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--collect-submodules",
        "pawahara_harness",
        "--collect-submodules",
        "e2b_code_interpreter",
        str(repo_root / "src" / "pawahara_harness" / "__main__.py"),
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
