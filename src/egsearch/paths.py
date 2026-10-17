from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
EMBEDDER_VERSION = "2"
NATIVE_DIM = 768
TRUNCATION_DIMS = (128, 256, 512, 768)


def project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("EGSEARCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "egsearch").is_dir():
            return candidate
    return here


def runtime_dir(root: Path) -> Path:
    path = root / ".egsearch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_path(root: Path) -> Path:
    return runtime_dir(root) / "index.sqlite"


def worker_socket(root: Path, role: str) -> Path:
    return runtime_dir(root) / f"{role}.sock"


def worker_pidfile(root: Path, role: str) -> Path:
    return runtime_dir(root) / f"{role}.pid"


def worker_log(root: Path, role: str) -> Path:
    return runtime_dir(root) / f"{role}.log"


def server_socket(root: Path) -> Path:
    return worker_socket(root, "search")


def server_pidfile(root: Path) -> Path:
    return worker_pidfile(root, "search")


def server_log(root: Path) -> Path:
    return worker_log(root, "search")


def model_cache(root: Path) -> Path:
    return root / "models"


def default_corpus(root: Path) -> Path:
    return root / "corpus"
