"""Names that show up in Activity Monitor.

Activity Monitor's Process Name is the executable's name (ucomm), which is
fixed when the process starts. A Python program therefore shows up as
python3.13 unless it re-execs a hard link of itself with a short name.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_MAX_NAME = 16
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def process_name(command: str, *, role: str = "search", paths: list[Path] | None = None) -> str:
    if command == "serve":
        if role == "index-0":
            return "eg-index"
        if role == "search":
            return "eg-search"
        return fit(f"eg-{role}")
    if command == "index":
        chosen = list(paths or [])
        if len(chosen) == 1:
            return fit(f"eg-{folder_label(chosen[0])}")
        if len(chosen) > 1:
            return fit(f"eg-{len(chosen)}dirs")
        return "eg-index"
    if command == "search":
        return "eg-query"
    return fit(f"eg-{command}")


def folder_label(path: Path) -> str:
    expanded = path.expanduser()
    name = expanded.name
    if name in {"", ".", ".."}:
        name = expanded.resolve().name or "index"
    slug = _SAFE.sub("-", name).strip("-").lower()
    return slug or "index"


def fit(name: str) -> str:
    cleaned = _SAFE.sub("-", name).strip("-")
    return (cleaned or "egsearch")[:_MAX_NAME]


def adopt_process_name(root: Path, name: str) -> None:
    """Re-exec this process so Activity Monitor shows `name`."""
    name = fit(name)
    if Path(sys.executable).name == name:
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        executable = named_executable(root, name)
    except OSError:
        return
    os.execv(str(executable), [name, *sys.argv])


def named_executable(root: Path, name: str) -> Path:
    # Activity Monitor shows this filename. The link has to sit in the
    # virtualenv's bin directory so Python still finds pyvenv.cfg.
    del root
    invoked = Path(sys.executable)
    source = invoked.resolve()
    destination = invoked.parent / fit(name)
    if destination.resolve() == source:
        raise OSError(f"refusing to replace the Python binary with {destination.name}")
    if destination.exists() or destination.is_symlink():
        try:
            if destination.samefile(source):
                return destination
        except OSError:
            pass
        destination.unlink()
    os.link(source, destination)
    return destination
