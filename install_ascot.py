"""
Installer utility for ASCOT5 (a5py) as part of alpha_analysis optional setup.

This script automates:
- Cloning https://github.com/ascot4fusion/ascot5
- Optionally checking out a branch or tag
- Installing its Python package and dependencies
- Building the native library with `make libascot` (honors CC if provided)

Usage (console script):
    alpha-analysis-install-ascot [--repo URL] [--branch BRANCH] [--dest PATH] [--cc CC]

Environment variables:
    CC: overrides the C compiler used by `make libascot` if not provided via --cc.

No network calls or builds happen unless you run this script explicitly.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "https://github.com/ascot4fusion/ascot5.git"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[alpha-analysis] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def clone_or_update(repo_url: str, dest: Path) -> None:
    if dest.exists() and (dest / ".git").exists():
        run(["git", "-C", str(dest), "remote", "set-url", "origin", repo_url])
        run(["git", "-C", str(dest), "fetch", "--all", "--tags", "--prune"])
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(dest)])


def checkout(dest: Path, ref: str | None) -> None:
    if ref:
        run(["git", "-C", str(dest), "checkout", ref])
        run(["git", "-C", str(dest), "pull", "--ff-only"])  # best effort


def pip_install_editable(dest: Path) -> None:
    # Ensure modern build tooling, then install the package in editable mode
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-e", str(dest)])


def make_libascot(dest: Path, cc: str | None) -> None:
    env = os.environ.copy()
    if cc:
        env["CC"] = cc
    print("[alpha-analysis] Building ASCOT native library with 'make libascot' ...")
    run(["make", "libascot"], cwd=dest, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install ASCOT5 (a5py) for alpha_analysis")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Git repository URL for ascot5")
    parser.add_argument("--branch", default=None, help="Branch, tag, or commit to checkout")
    parser.add_argument(
        "--dest",
        default=str(Path.cwd() / "no_sync" / "ascot5-src"),
        help="Destination directory to clone the repository into",
    )
    parser.add_argument("--cc", default=None, help="C compiler to use for building libascot")
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()

    print(f"[alpha-analysis] Using repository: {args.repo}")
    print(f"[alpha-analysis] Destination: {dest}")
    if args.branch:
        print(f"[alpha-analysis] Checkout ref: {args.branch}")

    # 1) Clone or update repository
    clone_or_update(args.repo, dest)

    # 2) Checkout requested ref (if any)
    checkout(dest, args.branch)

    # 3) Install Python package and dependencies
    pip_install_editable(dest)

    # 4) Build native library (honor CC)
    cc = args.cc or os.environ.get("CC")
    make_libascot(dest, cc)

    print("[alpha-analysis] ASCOT installation and build steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
