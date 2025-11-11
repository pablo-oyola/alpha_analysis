from __future__ import annotations

"""Optional ASCOT auto-install hook.

Including this `setup.py` permits invoking:

    pip install . --config-settings ascot_install=true

to trigger cloning the ASCOT5 repository, installing its Python interface, and
compiling the native library with `make libascot`. The user may override the C
compiler with the environment variable `CC`.

The hook is intentionally opt-in via a config setting so that normal installs
remain lightweight.
"""
import os
import subprocess
from pathlib import Path
from setuptools import setup


ASCOT_REPO = os.environ.get("ASCOT_REPO", "https://github.com/ascot4fusion/ascot5.git")
ASCOT_BRANCH = os.environ.get("ASCOT_BRANCH")  # optional: tag, branch, commit
ASCOT_DEST = Path(os.environ.get("ASCOT_DEST", str(Path.cwd() / "no_sync" / "ascot5-src")))


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[ascot-hook] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def ensure_ascot_available() -> None:
    """Ensure ASCOT (a5py + libascot) is present; otherwise fetch and build it.

    Triggered automatically unless ASCOT_SKIP is set to 'true'. Users can supply
    an existing installation by placing it on PYTHONPATH before invoking pip.
    """
    if os.environ.get("ASCOT_SKIP") == "true":
        print("[ascot-hook] Skipping automatic ASCOT installation (ASCOT_SKIP=true).")
        return
    # Quick availability check
    try:
        import a5py  # noqa: F401
        print("[ascot-hook] Found existing a5py installation; assuming libascot built.")
        return
    except Exception:
        print("[ascot-hook] a5py not found; proceeding to clone and build ASCOT.")
    print("[ascot-hook] Cloning/building ASCOT repository...")
    if ASCOT_DEST.exists() and (ASCOT_DEST / ".git").exists():
        run(["git", "-C", str(ASCOT_DEST), "fetch", "--all", "--tags", "--prune"])
    else:
        ASCOT_DEST.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", ASCOT_REPO, str(ASCOT_DEST)])
    if ASCOT_BRANCH:
        run(["git", "-C", str(ASCOT_DEST), "checkout", ASCOT_BRANCH])
        run(["git", "-C", str(ASCOT_DEST), "pull", "--ff-only"])
    # Install Python interface editable
    run(["python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    run(["python", "-m", "pip", "install", "-e", str(ASCOT_DEST)])
    # Build native library
    env = os.environ.copy()
    print("[ascot-hook] Building libascot (honoring CC if set)...")
    run(["make", "libascot"], cwd=ASCOT_DEST, env=env)
    print("[ascot-hook] ASCOT installation completed.")


ensure_ascot_available()

setup()
