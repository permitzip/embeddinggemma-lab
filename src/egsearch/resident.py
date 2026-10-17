from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from egsearch.paths import runtime_dir, worker_log, worker_pidfile, worker_socket

_MAX_MESSAGE = 8_000_000
_READY_TIMEOUT = 180.0


class ResidentError(RuntimeError):
    pass


def call(root: Path, payload: dict, timeout: float = 180.0, role: str = "search") -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(str(worker_socket(root, role)))
        except OSError as exc:
            raise ResidentError(f"{role} process is not running") from exc
        _send(sock, payload)
        reply = _recv(sock)
    if "error" in reply:
        raise ResidentError(str(reply["error"]))
    return reply


def running(root: Path, role: str = "search") -> bool:
    pid = _read_pid(root, role)
    if pid is not None and _pid_running(pid):
        return True
    try:
        return call(root, {"op": "ping"}, timeout=2, role=role).get("ok") is True
    except (ResidentError, TimeoutError, OSError):
        return False


def ensure_worker(root: Path, role: str) -> dict:
    """Return ping info, starting a detached process for this role when none is loaded."""
    import fcntl

    if _role_alive(root, role):
        return call(root, {"op": "ping"}, timeout=_READY_TIMEOUT, role=role)
    lock_path = runtime_dir(root) / "server.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _role_alive(root, role):
            return call(root, {"op": "ping"}, timeout=_READY_TIMEOUT, role=role)
        _clear_stale(root, role)
        log_path = worker_log(root, role)
        print(f"Loading EmbeddingGemma 2 ({role})…", file=sys.stderr)
        with log_path.open("a") as log:
            process = subprocess.Popen(
                _serve_command(role),
                cwd=str(root),
                env={**os.environ, "EGSEARCH_ROOT": str(root)},
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + _READY_TIMEOUT
        while time.monotonic() < deadline:
            if running(root, role) and _ping_now(root, role):
                return call(root, {"op": "ping"}, timeout=5, role=role)
            if process.poll() is not None:
                raise ResidentError(f"{role} process exited. See {log_path}")
            time.sleep(0.2)
        raise ResidentError(f"{role} process did not become ready. See {log_path}")


def ensure_workers(root: Path, roles: list[str]) -> dict[str, dict]:
    import fcntl

    lock_path = runtime_dir(root) / "server.lock"
    started: list[tuple[str, subprocess.Popen]] = []
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for role in roles:
            if _role_alive(root, role):
                continue
            _clear_stale(root, role)
            log_path = worker_log(root, role)
            print(f"Loading EmbeddingGemma 2 ({role})…", file=sys.stderr)
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    _serve_command(role),
                    cwd=str(root),
                    env={**os.environ, "EGSEARCH_ROOT": str(root)},
                    start_new_session=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            started.append((role, process))
        infos: dict[str, dict] = {}
        deadline = time.monotonic() + _READY_TIMEOUT
        for role in roles:
            while time.monotonic() < deadline:
                if _ping_now(root, role):
                    infos[role] = call(root, {"op": "ping"}, timeout=5, role=role)
                    break
                failed = next(
                    (process for name, process in started if name == role and process.poll() is not None),
                    None,
                )
                if failed is not None:
                    raise ResidentError(f"{role} process exited. See {worker_log(root, role)}")
                time.sleep(0.2)
            else:
                raise ResidentError(f"{role} process did not become ready. See {worker_log(root, role)}")
        return infos


def stop(root: Path) -> bool:
    roles = _known_roles(root)
    if not roles:
        return False
    stopped = False
    for role in roles:
        if _stop_role(root, role):
            stopped = True
    return stopped


def _stop_role(root: Path, role: str) -> bool:
    if not running(root, role):
        _clear_stale(root, role)
        return False
    try:
        call(root, {"op": "shutdown"}, timeout=30, role=role)
    except ResidentError:
        pid = _read_pid(root, role)
        if pid is not None and _pid_running(pid):
            os.kill(pid, 15)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not _role_alive(root, role):
            _clear_stale(root, role)
            return True
        time.sleep(0.1)
    raise ResidentError(f"{role} process did not exit")


def run(root: Path, embedder, role: str = "search") -> None:
    import signal

    sock_path = worker_socket(root, role)
    _clear_stale(root, role)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    server.listen(8)
    worker_pidfile(root, role).write_text(str(os.getpid()), encoding="utf-8")
    stop_requested = False

    def _handle_term(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        server.close()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)
    try:
        while not stop_requested:
            try:
                connection, _address = server.accept()
            except OSError:
                break
            with connection:
                try:
                    request = _recv(connection)
                    reply, should_stop = _dispatch(embedder, request, role)
                    _send(connection, reply)
                    stop_requested = should_stop
                except Exception as exc:
                    try:
                        _send(connection, {"error": str(exc)})
                    except OSError:
                        pass
    finally:
        server.close()
        sock_path.unlink(missing_ok=True)
        worker_pidfile(root, role).unlink(missing_ok=True)


def _dispatch(embedder, request: dict, role: str) -> tuple[dict, bool]:
    op = request.get("op")
    if op == "ping":
        return {
            "ok": True,
            "pid": os.getpid(),
            "role": role,
            "device": str(getattr(embedder, "device", getattr(embedder.model, "device", "unknown"))),
            "dim": int(embedder.dim),
            "parameters": _parameter_count(embedder),
        }, False
    if op == "shutdown":
        return {"ok": True}, True
    if op == "embed_query":
        vector = embedder.embed_query(
            text=str(request.get("text") or ""),
            image=_path(request.get("image")),
            video=_path(request.get("video")),
            audio=_path(request.get("audio")),
            task=str(request.get("task") or "search"),
        )
        return {"vector": _vector(vector)}, False
    if op == "embed_text_document":
        vector = embedder.embed_text_document(str(request["title"]), str(request["body"]))
        return {"vector": _vector(vector)}, False
    if op == "embed_text_query":
        vector = embedder.embed_text_query(str(request["text"]), str(request.get("task") or "search"))
        return {"vector": _vector(vector)}, False
    if op == "embed_image":
        vector = embedder.embed_image(Path(str(request["path"])))
        return {"vector": _vector(vector)}, False
    if op == "embed_video_window":
        from egsearch.index import _embed_video

        vector = _embed_video(
            embedder,
            Path(str(request["path"])),
            float(request["start"]),
            float(request["length"]),
        )
        return {"vector": _vector(vector)}, False
    if op == "embed_audio_window":
        from egsearch.index import _embed_audio

        vector = _embed_audio(
            embedder,
            Path(str(request["path"])),
            float(request["start"]),
            float(request["length"]),
        )
        return {"vector": _vector(vector)}, False
    if op == "embed_batch":
        vectors = embedder.embed_batch(list(request["items"]))
        return {"vectors": [_vector(vector) for vector in vectors]}, False
    if op == "embed_pdf_page":
        from egsearch.index import _embed_pdf

        vector = _embed_pdf(
            embedder,
            Path(str(request["path"])),
            int(request["page"]),
            str(request["title"]),
            str(request["excerpt"]),
        )
        return {"vector": _vector(vector)}, False
    raise ResidentError(f"unknown operation {op}")


class _RpcEmbedder:
    dim: int

    def _rpc(self, payload: dict) -> dict:
        raise NotImplementedError

    def embed_query(self, **kwargs):
        return _as_vector(
            self._rpc(
                {
                    "op": "embed_query",
                    "text": kwargs.get("text") or "",
                    "image": _str_path(kwargs.get("image")),
                    "video": _str_path(kwargs.get("video")),
                    "audio": _str_path(kwargs.get("audio")),
                    "task": kwargs.get("task") or "search",
                }
            )
        )

    def embed_text_document(self, title: str, body: str):
        return _as_vector(self._rpc({"op": "embed_text_document", "title": title, "body": body}))

    def embed_text_query(self, text: str, task: str):
        return _as_vector(self._rpc({"op": "embed_text_query", "text": text, "task": task}))

    def embed_image(self, path: Path):
        return _as_vector(self._rpc({"op": "embed_image", "path": str(path)}))

    def embed_video_window(self, path: Path, start: float, length: float):
        return _as_vector(
            self._rpc(
                {"op": "embed_video_window", "path": str(path), "start": start, "length": length}
            )
        )

    def embed_audio_window(self, path: Path, start: float, length: float):
        return _as_vector(
            self._rpc(
                {"op": "embed_audio_window", "path": str(path), "start": start, "length": length}
            )
        )

    def embed_pdf_page(self, path: Path, page: int, title: str, excerpt: str):
        return _as_vector(
            self._rpc(
                {
                    "op": "embed_pdf_page",
                    "path": str(path),
                    "page": page,
                    "title": title,
                    "excerpt": excerpt,
                }
            )
        )


class RemoteEmbedder(_RpcEmbedder):
    """Talks only to the search process, which does not index files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        info = ensure_worker(root, "search")
        self.dim = int(info["dim"])

    def _rpc(self, payload: dict) -> dict:
        return call(self.root, payload, role="search")


class IndexEmbedder(_RpcEmbedder):
    """One GPU process. Callers prepare files on CPU and send batches."""

    def __init__(self, root: Path, workers: int = 1) -> None:
        del workers
        self.root = root
        info = ensure_worker(root, "index-0")
        self.dim = int(info["dim"])
        self._lock = threading.Lock()

    def _rpc(self, payload: dict) -> dict:
        with self._lock:
            return call(self.root, payload, role="index-0")

    def embed_batch(self, items: list[dict]) -> list[list[float]]:
        reply = self._rpc({"op": "embed_batch", "items": items})
        return [[float(value) for value in vector] for vector in reply["vectors"]]


def _parameter_count(embedder) -> int:
    model = getattr(embedder, "model", None)
    if model is None:
        return 0
    return sum(parameter.numel() for parameter in model.parameters())


def _vector(values) -> list[float]:
    return [float(value) for value in values]


def _as_vector(reply: dict) -> list[float]:
    return [float(value) for value in reply["vector"]]


def _path(value) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _str_path(value) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _serve_command(role: str) -> list[str]:
    invoked = os.path.basename(sys.argv[0])
    if invoked == "egsearch":
        return [sys.argv[0], "serve", "--role", role]
    return [sys.executable, "-m", "egsearch", "serve", "--role", role]


def _known_roles(root: Path) -> list[str]:
    return sorted(path.stem for path in runtime_dir(root).glob("*.pid"))


def _role_alive(root: Path, role: str) -> bool:
    pid = _read_pid(root, role)
    return pid is not None and _pid_running(pid)


def _ping_now(root: Path, role: str) -> bool:
    try:
        return call(root, {"op": "ping"}, timeout=1, role=role).get("ok") is True
    except (ResidentError, TimeoutError, OSError):
        return False


def _clear_stale(root: Path, role: str) -> None:
    pid = _read_pid(root, role)
    if pid is not None and _pid_running(pid):
        return
    worker_socket(root, role).unlink(missing_ok=True)
    worker_pidfile(root, role).unlink(missing_ok=True)


def _read_pid(root: Path, role: str = "search") -> int | None:
    path = worker_pidfile(root, role)
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _send(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    if len(body) > _MAX_MESSAGE:
        raise ResidentError("message is too large")
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv(sock: socket.socket) -> dict:
    raw_length = _read_exact(sock, 4)
    length = struct.unpack(">I", raw_length)[0]
    if length > _MAX_MESSAGE:
        raise ResidentError("message is too large")
    return json.loads(_read_exact(sock, length).decode("utf-8"))


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        block = sock.recv(remaining)
        if not block:
            raise ResidentError("connection closed")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)
