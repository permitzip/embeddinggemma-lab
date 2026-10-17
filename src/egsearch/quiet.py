from __future__ import annotations

import logging
import os
import sys
import warnings

_NOISE = (
    "Redirects are currently not supported",
    "unauthenticated requests to the HF Hub",
    "set a HF_TOKEN",
    "Loading weights:",
    "Image decoding with",
    "torchcodec",
    "Asked to sample uniformly",
    "video metadata has no",
    "Syntax Error",
    "Syntax Warning",
    "Illegal character",
)


class _QuietStderr:
    def __init__(self, underlying):
        self.underlying = underlying
        self.pending = ""

    def write(self, data: str) -> int:
        self.pending += data
        while True:
            split_at = _next_break(self.pending)
            if split_at is None:
                break
            line = self.pending[:split_at]
            separator = self.pending[split_at]
            self.pending = self.pending[split_at + 1 :]
            if _is_noise(line):
                continue
            self.underlying.write(line + separator)
        return len(data)

    def flush(self) -> None:
        if self.pending and not _is_noise(self.pending):
            self.underlying.write(self.pending)
        self.pending = ""
        self.underlying.flush()

    def isatty(self) -> bool:
        return self.underlying.isatty()

    def fileno(self) -> int:
        return self.underlying.fileno()


def install() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    for name in (
        "huggingface_hub",
        "transformers",
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r".*Redirects are currently not supported.*")
    warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*")
    warnings.filterwarnings("ignore", message=r".*torchcodec.*")
    warnings.filterwarnings("ignore", message=r".*Image decoding with.*")
    if not isinstance(sys.stderr, _QuietStderr):
        sys.stderr = _QuietStderr(sys.stderr)


def prefix_logs(name: str) -> None:
    """Put the process name at the start of each stderr line."""
    current = sys.stderr
    if isinstance(current, _QuietStderr):
        if not isinstance(current.underlying, _LinePrefix):
            current.underlying = _LinePrefix(current.underlying, name)
        return
    if not isinstance(current, _LinePrefix):
        sys.stderr = _LinePrefix(current, name)


class _LinePrefix:
    def __init__(self, underlying, name: str) -> None:
        self.underlying = underlying
        self._prefix = f"{name}  "
        self._at_line_start = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        pieces: list[str] = []
        for line in data.splitlines(keepends=True):
            if self._at_line_start:
                pieces.append(self._prefix)
            pieces.append(line)
            self._at_line_start = line.endswith(("\n", "\r"))
        self.underlying.write("".join(pieces))
        return len(data)

    def flush(self) -> None:
        self.underlying.flush()

    def isatty(self) -> bool:
        return self.underlying.isatty()

    def fileno(self) -> int:
        return self.underlying.fileno()


def _next_break(text: str) -> int | None:
    candidates = [index for index in (text.find("\n"), text.find("\r")) if index != -1]
    if not candidates:
        return None
    return min(candidates)


def _is_noise(line: str) -> bool:
    return any(snippet in line for snippet in _NOISE)
