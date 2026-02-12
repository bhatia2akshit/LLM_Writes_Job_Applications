from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike[str] = ".env.local", *, override: bool = False) -> None:
    """
    Lightweight dotenv loader (supports KEY=VALUE, comments, and quoted values).
    Does not require external dependencies.
    """
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def load_env_local(*, override: bool = False) -> None:
    """
    Load .env.local from backend/ first, then repo root if present.
    """
    here = Path(__file__).resolve().parent
    candidates = [here / ".env.local", here.parent / ".env.local"]
    for candidate in candidates:
        load_env_file(candidate, override=override)
